"""Paged KVarN-MLA decode kernel: packed latent in a paged cache
[num_blocks, PAGE, record], indexed via block_table. Validated vs fp16 ref.
record = [ 256 packed latent | scale fp16 | zp fp16 | 64 rope fp16 ] = 388 B.
"""
import torch, triton, triton.language as tl

H, L, RP, BITS = 16, 512, 64, 4
QMAX = (1 << BITS) - 1
NB = L * BITS // 8                                   # 256
REC = NB + 2 + 2 + RP * 2                            # 388 bytes/token
SCALE_OFF, ZP_OFF, ROPE_OFF = NB, NB + 2, NB + 4
PAGE = 64


def quant_token(latent):
    lo = latent.amin(1, keepdim=True); hi = latent.amax(1, keepdim=True)
    scale = ((hi - lo) / QMAX).clamp_min(1e-8); zp = lo
    q = torch.clamp(torch.round((latent - zp) / scale), 0, QMAX).to(torch.uint8)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()
    return packed, scale.squeeze(1), zp.squeeze(1)


def build_cache(latent, rope, block_table, seq_len, num_blocks):
    """Scatter S tokens into a paged uint8 cache by block_table."""
    S = latent.shape[0]
    packed, scale, zp = quant_token(latent)
    cache = torch.zeros(num_blocks, PAGE, REC, dtype=torch.uint8, device=latent.device)
    for i in range(S):
        phys = block_table[i // PAGE].item() * PAGE + i % PAGE
        bk, sl = phys // PAGE, phys % PAGE
        cache[bk, sl, :NB] = packed[i]
        cache[bk, sl, SCALE_OFF:SCALE_OFF + 2] = scale[i:i+1].to(torch.float16).view(torch.uint8)
        cache[bk, sl, ZP_OFF:ZP_OFF + 2] = zp[i:i+1].to(torch.float16).view(torch.uint8)
        cache[bk, sl, ROPE_OFF:] = rope[i].to(torch.float16).view(torch.uint8)
    return cache, packed, scale, zp


def ref_attn(q_lat, q_rope, packed, scale, zp, rope, sm, S):
    qd = torch.empty(S, L, dtype=torch.float32, device=q_lat.device)
    qd[:, 0::2] = (packed & 0xF).float(); qd[:, 1::2] = (packed >> 4).float()
    lat = qd * scale[:, None] + zp[:, None]
    qk = (q_lat @ lat.t() + q_rope @ rope.t()) * sm
    return torch.softmax(qk, 1) @ lat


@triton.jit
def _paged(Q_lat, Q_rope, Cache, BlockTable, Out, sm_scale, seq_len,
          L: tl.constexpr, RP: tl.constexpr, NB: tl.constexpr, REC: tl.constexpr,
          SCALE_OFF: tl.constexpr, ZP_OFF: tl.constexpr, ROPE_OFF: tl.constexpr,
          PAGE: tl.constexpr):
    h = tl.program_id(0)
    offs_l = tl.arange(0, L); offs_p = tl.arange(0, NB); offs_r = tl.arange(0, RP)
    q_lat = tl.load(Q_lat + h * L + offs_l).to(tl.float32)
    q_rope = tl.load(Q_rope + h * RP + offs_r).to(tl.float32)
    e_max = -float("inf"); e_sum = 0.0
    acc = tl.zeros([L], dtype=tl.float32)
    for s in range(0, seq_len):
        phys_blk = tl.load(BlockTable + s // PAGE)
        base = (phys_blk * PAGE + s % PAGE) * REC
        b = tl.load(Cache + base + offs_p).to(tl.uint32)
        sc = tl.load((Cache + base + SCALE_OFF).to(tl.pointer_type(tl.float16))).to(tl.float32)
        zp = tl.load((Cache + base + ZP_OFF).to(tl.pointer_type(tl.float16))).to(tl.float32)
        lat = tl.interleave((b & 0xF).to(tl.float32) * sc + zp,
                            ((b >> 4) & 0xF).to(tl.float32) * sc + zp)
        rp = tl.load((Cache + base + ROPE_OFF).to(tl.pointer_type(tl.float16)) + offs_r).to(tl.float32)
        qk = (tl.sum(q_lat * lat) + tl.sum(q_rope * rp)) * sm_scale
        new_max = tl.maximum(e_max, qk)
        p = tl.exp(qk - new_max); alpha = tl.exp(e_max - new_max)
        e_sum = e_sum * alpha + p
        acc = acc * alpha + p * lat
        e_max = new_max
    tl.store(Out + h * L + offs_l, acc / e_sum)


def main():
    torch.manual_seed(0); dev = "cuda"
    S = 200
    num_blocks = 16
    # shuffled block_table to exercise paging indirection
    perm = torch.randperm(num_blocks, device=dev).to(torch.int32)
    nblk = (S + PAGE - 1) // PAGE
    block_table = perm[:nblk]
    q_lat = torch.randn(H, L, device=dev); q_rope = torch.randn(H, RP, device=dev)
    latent = (torch.randn(S, L, device=dev) * 0.5 + torch.randn(1, L, device=dev) * 2)
    rope = torch.randn(S, RP, device=dev)
    sm = 1.0 / (L ** 0.5)
    cache, packed, scale, zp = build_cache(latent, rope, block_table, S, num_blocks)
    ref = ref_attn(q_lat, q_rope, packed, scale, zp, rope, sm, S)
    out = torch.zeros(H, L, device=dev)
    _paged[(H,)](q_lat, q_rope, cache, block_table, out, sm, S,
                 L=L, RP=RP, NB=NB, REC=REC, SCALE_OFF=SCALE_OFF, ZP_OFF=ZP_OFF,
                 ROPE_OFF=ROPE_OFF, PAGE=PAGE)
    cos = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), 0).item()
    mx = (out - ref).abs().max().item()
    print(f"paged kernel vs fp16-ref: cos={cos:.6f}  max_abs={mx:.2e}")
    print("PAGED_OK" if cos > 0.9999 and mx < 1e-2 else "PAGED_MISMATCH")


if __name__ == "__main__":
    main()
