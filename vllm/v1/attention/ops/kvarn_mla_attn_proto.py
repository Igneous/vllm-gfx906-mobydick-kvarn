"""Minimal KVarN-MLA decode attention with in-kernel 4-bit dequant of the
latent. Proves the core math: load packed latent -> unpack -> dequant ->
use as both K and V (MLA) + fp16 rope, vs an fp16 reference.
v1: per-token asymmetric RTN, no Hadamard.
"""
import torch, triton, triton.language as tl

S, H, L, RP, BITS = 256, 16, 512, 64, 4
QMAX = (1 << BITS) - 1
NB = L * BITS // 8                                  # 256 packed bytes


def make_packed(latent):
    lo = latent.amin(1, keepdim=True); hi = latent.amax(1, keepdim=True)
    scale = ((hi - lo) / QMAX).clamp_min(1e-8); zp = lo
    q = torch.clamp(torch.round((latent - zp) / scale), 0, QMAX).to(torch.uint8)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()          # [S,NB]
    return packed, scale.squeeze(1).contiguous(), zp.squeeze(1).contiguous()


def ref_attn(q_lat, q_rope, latent, rope, sm):
    packed, scale, zp = make_packed(latent)
    qd = torch.empty(S, L, dtype=torch.float32, device=latent.device)
    qd[:, 0::2] = (packed & 0xF).float(); qd[:, 1::2] = (packed >> 4).float()
    lat_dq = qd * scale[:, None] + zp[:, None]
    qk = (q_lat @ lat_dq.t() + q_rope @ rope.t()) * sm
    p = torch.softmax(qk, dim=1)
    return p @ lat_dq


@triton.jit
def _kvarn_mla_attn(Q_lat, Q_rope, Packed, Scale, Zp, Rope, Out, sm_scale,
                    S: tl.constexpr, L: tl.constexpr, RP: tl.constexpr, NB: tl.constexpr):
    h = tl.program_id(0)
    offs_l = tl.arange(0, L)
    offs_p = tl.arange(0, NB)
    offs_r = tl.arange(0, RP)
    q_lat = tl.load(Q_lat + h * L + offs_l).to(tl.float32)
    q_rope = tl.load(Q_rope + h * RP + offs_r).to(tl.float32)
    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([L], dtype=tl.float32)
    for s in range(0, S):
        b = tl.load(Packed + s * NB + offs_p).to(tl.uint32)
        sc = tl.load(Scale + s).to(tl.float32)
        zp = tl.load(Zp + s).to(tl.float32)
        lat_lo = (b & 0xF).to(tl.float32) * sc + zp                 # even channels
        lat_hi = ((b >> 4) & 0xF).to(tl.float32) * sc + zp          # odd channels
        lat = tl.interleave(lat_lo, lat_hi)                         # [L]
        rp = tl.load(Rope + s * RP + offs_r).to(tl.float32)
        qk = (tl.sum(q_lat * lat) + tl.sum(q_rope * rp)) * sm_scale
        new_max = tl.maximum(e_max, qk)
        p = tl.exp(qk - new_max)
        alpha = tl.exp(e_max - new_max)
        e_sum = e_sum * alpha + p
        acc = acc * alpha + p * lat
        e_max = new_max
    tl.store(Out + h * L + offs_l, acc / e_sum)


def main():
    torch.manual_seed(0); dev = "cuda"
    q_lat = torch.randn(H, L, device=dev); q_rope = torch.randn(H, RP, device=dev)
    latent = (torch.randn(S, L, device=dev) * 0.5 + torch.randn(1, L, device=dev) * 2)
    rope = torch.randn(S, RP, device=dev)
    sm = 1.0 / (L ** 0.5)
    ref = ref_attn(q_lat, q_rope, latent, rope, sm)
    packed, scale, zp = make_packed(latent)
    out = torch.zeros(H, L, device=dev)
    _kvarn_mla_attn[(H,)](q_lat, q_rope, packed, scale, zp, rope, out, sm,
                          S=S, L=L, RP=RP, NB=NB)
    cos = torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), 0).item()
    mx = (out - ref).abs().max().item()
    print(f"kernel vs fp16-ref: cos={cos:.6f}  max_abs={mx:.2e}")
    print("KERNEL_OK" if cos > 0.9999 and mx < 1e-2 else "KERNEL_MISMATCH")


if __name__ == "__main__":
    main()
