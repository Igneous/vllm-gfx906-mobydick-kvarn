# SPDX-License-Identifier: Apache-2.0
"""LOCAL EXPERIMENT (not for upstream): KVarN quantization round-trip on the
MLA compressed latent, to measure the *accuracy* impact of applying KVarN's
recipe (Hadamard rotation -> log-domain Sinkhorn variance normalization ->
asymmetric RTN) to DeepSeek-style MLA latents.

It does a full round trip (rotate -> sinkhorn -> RTN -> dequant -> inverse
sinkhorn -> un-rotate) on the [T, kv_lora_rank] latent and returns a lossy
fp version. The MLA attention downstream is unchanged, so this isolates the
quantization error without needing weight absorption or a custom kernel /
cache layout. No memory savings here — this is an accuracy probe only.

Enabled per-process via env KVARN_MLA_BITS (e.g. 4 or 2); unset = passthrough.
"""
import functools
import os

import torch

from vllm.model_executor.layers.quantization.kvarn.sinkhorn import (
    variance_normalize,
)


def mla_bits() -> int:
    return int(os.environ.get("KVARN_MLA_BITS", "0"))


@functools.lru_cache(maxsize=8)
def _hadamard(n: int, device_str: str) -> torch.Tensor:
    """Orthonormal Sylvester-Hadamard of size n (n must be a power of 2)."""
    assert (n & (n - 1)) == 0, f"Hadamard size {n} must be a power of 2"
    H = torch.ones(1, 1, dtype=torch.float32)
    while H.shape[0] < n:
        H = torch.cat(
            [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0
        )
    H = H / (n ** 0.5)
    return H.to(device_str)


def kvarn_mla_roundtrip(latent: torch.Tensor, bits: int, group: int = 128) -> torch.Tensor:
    """latent: [T, R] (R = kv_lora_rank). Returns lossy [T, R] same dtype.

    Mirrors KVarN's K path: rotate channels, then for each block of `group`
    tokens balance the [R, g] tile (per-channel x per-token std-norm) and
    asymmetric-RTN per channel, dequant, invert the balance, un-rotate.
    """
    if bits <= 0:
        return latent
    if not getattr(kvarn_mla_roundtrip, "_logged", False):
        kvarn_mla_roundtrip._logged = True
        print(f"[KVARN-MLA] probe ACTIVE: bits={bits} on latent {tuple(latent.shape)}",
              flush=True)
    # one-shot dump of a REAL latent (skip the zero-variance profile-run dummy)
    _dump = os.environ.get("KVARN_MLA_DUMP")
    _flat = latent.reshape(-1, latent.shape[-1]).float()
    # token-variance (per-channel std across tokens): ~0 for vLLM's repeated
    # profile-dummy token, >0 for a real prompt. Gate on it to skip the dummy.
    _tokvar = _flat.std(0).mean().item() if _flat.shape[0] > 16 else 0.0
    if (_dump and not getattr(kvarn_mla_roundtrip, "_dumped", False)
            and _tokvar > 0.01):
        kvarn_mla_roundtrip._dumped = True
        torch.save(_flat.detach().cpu(), _dump)
        print(f"[KVARN-MLA] dumped real latent {tuple(_flat.shape)} "
              f"tokvar={_tokvar:.3f} to {_dump}", flush=True)
    orig_shape = latent.shape
    R = orig_shape[-1]
    x = latent.reshape(-1, R).float()
    T = x.shape[0]
    H = _hadamard(R, str(latent.device))
    xr = x @ H                                   # rotate channels
    qmax = (1 << bits) - 1
    out = torch.empty_like(xr)
    for s in range(0, T, group):
        tile = xr[s:s + group]                   # [g, R]
        t = tile.t().contiguous()                # [R, g]  (channels x tokens)
        bal, s_col, s_row = variance_normalize(t)  # bal = t / s_col / s_row
        lo = bal.amin(dim=1, keepdim=True)
        hi = bal.amax(dim=1, keepdim=True)
        scale = ((hi - lo) / qmax).clamp_min(1e-8)
        q = torch.clamp(torch.round((bal - lo) / scale), 0, qmax)
        deq = q * scale + lo
        rec = deq * s_col * s_row                # invert sinkhorn -> ~[R, g]
        out[s:s + group] = rec.t()
    xu = out @ H.t()                             # un-rotate
    return xu.reshape(orig_shape).to(latent.dtype)
