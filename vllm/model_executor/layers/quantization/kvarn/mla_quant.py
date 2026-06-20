# SPDX-License-Identifier: Apache-2.0
"""LOCAL EXPERIMENT: per-token KVarN quantization of the MLA latent.

Streaming-friendly (no tile pool): each token's kv_lora_rank latent is rotated
by a fixed Hadamard, asymmetric-RTN'd over its channels (one scale+zp per
token), and packed. The decoupled RoPE part is kept fp16. Packed per-token
record (uint8 cache, head_size = bytes):

    [ packed latent : R*bits/8 ][ scale : 2 ][ zp : 2 ][ rope : Rope*2 ]

This trades Sinkhorn's variance balancing for streaming simplicity; Hadamard +
asymmetric RTN already spreads outliers (QuaRot-style). Used to get a runnable
KVarN-MLA backend and a real burst number; Sinkhorn is a later refinement.
"""
import functools

import torch


@functools.lru_cache(maxsize=8)
def hadamard(n: int, device_str: str) -> torch.Tensor:
    H = torch.ones(1, 1, dtype=torch.float32)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / (n ** 0.5)).to(device_str)


def record_bytes(R: int, rope: int, bits: int) -> int:
    return (R * bits) // 8 + 2 + 2 + rope * 2


def pack_tokens(latent: torch.Tensor, rope: torch.Tensor, bits: int) -> torch.Tensor:
    """latent [T, R], rope [T, Rope] -> packed uint8 [T, record_bytes]."""
    T, R = latent.shape
    qmax = (1 << bits) - 1
    H = hadamard(R, str(latent.device))
    xr = latent.float() @ H                                  # rotate
    lo = xr.amin(1, keepdim=True)
    hi = xr.amax(1, keepdim=True)
    scale = ((hi - lo) / qmax).clamp_min(1e-8)
    q = torch.clamp(torch.round((xr - lo) / scale), 0, qmax).to(torch.uint8)  # [T,R]
    per = 8 // bits
    packed = q[:, 0::2] | (q[:, 1::2] << 4) if bits == 4 else _packn(q, bits, per)  # [T, R/per]
    rec = torch.empty(T, record_bytes(R, rope.shape[1], bits),
                      dtype=torch.uint8, device=latent.device)
    nb = (R * bits) // 8
    rec[:, :nb] = packed
    rec[:, nb:nb + 2] = scale.to(torch.float16).view(torch.uint8)
    rec[:, nb + 2:nb + 4] = lo.to(torch.float16).view(torch.uint8)
    rec[:, nb + 4:] = rope.to(torch.float16).view(torch.uint8).reshape(T, -1)
    return rec


def _packn(q, bits, per):
    T, R = q.shape
    out = torch.zeros(T, R // per, dtype=torch.uint8, device=q.device)
    for i in range(per):
        out |= q[:, i::per] << (bits * i)
    return out


def unpack_tokens(rec: torch.Tensor, R: int, rope: int, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """packed uint8 [T, bytes] -> (latent [T,R] fp32 un-rotated, rope [T,Rope] fp16)."""
    T = rec.shape[0]
    nb = (R * bits) // 8
    per = 8 // bits
    packed = rec[:, :nb]
    scale = rec[:, nb:nb + 2].contiguous().view(torch.float16).float()       # [T,1]
    zp = rec[:, nb + 2:nb + 4].contiguous().view(torch.float16).float()      # [T,1]
    rope_t = rec[:, nb + 4:].contiguous().view(torch.float16).reshape(T, rope)
    q = torch.empty(T, R, dtype=torch.float32, device=rec.device)
    for i in range(per):
        q[:, i::per] = ((packed >> (bits * i)) & ((1 << bits) - 1)).float()
    xr = q * scale + zp
    H = hadamard(R, str(rec.device))
    latent = xr @ H.t()                                                       # un-rotate
    return latent, rope_t


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda"
    R, Rope, bits = 512, 64, 4
    lat = (torch.randn(300, R) * 0.5 + torch.randn(1, R) * 2).to(dev)
    rope = torch.randn(300, Rope).to(dev)
    rec = pack_tokens(lat, rope, bits)
    lat2, rope2 = unpack_tokens(rec, R, Rope, bits)
    cl = torch.nn.functional.cosine_similarity(lat.flatten(), lat2.flatten(), 0).item()
    cr = torch.nn.functional.cosine_similarity(rope.flatten().float(), rope2.flatten().float(), 0).item()
    print(f"record_bytes={rec.shape[1]} (vs fp16 {(R+Rope)*2}) "
          f"=> {(R+Rope)*2/rec.shape[1]:.2f}x  latent_cos={cl:.4f}  rope_cos={cr:.5f}")
    print("MLA_QUANT_OK" if cl > 0.99 and cr > 0.999 else "MLA_QUANT_FAIL")
