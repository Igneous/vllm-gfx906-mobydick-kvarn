# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import ClassVar

import torch

import vllm.envs as envs
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.triton_utils import triton
import triton.language as tl


def kvarn_mla_tile_layout(kv_lora_rank: int, rope_dim: int, group: int, bits: int):
    """Per-BLOCK tile record byte layout for the full-method KVarN-MLA cache
    (K-path: per-channel scale/zp + per-token s_row, packed token-major, fp16
    rope). One record = ``group`` tokens. Returns
    (NB, SC, ZP, SR, RP, REC, head_size) where head_size = REC // group."""
    R = kv_lora_rank
    nb = group * R * bits // 8          # packed latent
    sc = nb                             # scale[R] fp16 (per-channel)
    zp = sc + R * 2
    sr = zp + R * 2                     # s_row[group] fp16 (per-token)
    rp = sr + group * 2                 # rope group*rope_dim fp16
    rec = rp + group * rope_dim * 2
    assert rec % group == 0, f"tile rec {rec} not divisible by group {group}"
    return nb, sc, zp, sr, rp, rec, rec // group


def kvarn_mla_layout(kv_lora_rank: int, rope_dim: int, bits: int):
    """Packed-record byte layout for KVarN-MLA. Fields are 16-byte aligned so the
    decode kernel's gathered fp16 loads (scale/zp/rope, strided by REC) vectorize
    without misaligned-address faults. Returns (NB, scale_off, zp_off, rope_off, rec)."""
    NB = (kv_lora_rank * bits) // 8

    def au(x):
        return (x + 15) // 16 * 16

    scale_off = au(NB)
    zp_off = scale_off + 16
    rope_off = zp_off + 16
    rec = au(rope_off + rope_dim * 2)
    return NB, scale_off, zp_off, rope_off, rec


@triton.jit
def _kvarn_mla_decode_kernel(
    Q, Cache, BlockTable, Seqlens, O, Lse, sm_scale,
    stride_qb, stride_qh, stride_btb, stride_ob, stride_oh, stride_lb,
    L: tl.constexpr, RP: tl.constexpr, NB: tl.constexpr, REC: tl.constexpr,
    SCALE_OFF: tl.constexpr, ZP_OFF: tl.constexpr, ROPE_OFF: tl.constexpr,
    PAGE: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Batched paged KVarN-MLA decode: grid (B, H). One (sequence, head) per
    program; tiles the sequence's tokens in BLOCK_N chunks (vectorized, not the
    old per-token serial loop), dequants the packed latent in-kernel (4-bit +
    per-token scale/zp) as K and V, adds fp16 RoPE, online softmax. cos 1.0."""
    bcur = tl.program_id(0)
    h = tl.program_id(1)
    seq_len = tl.load(Seqlens + bcur)
    offs_l = tl.arange(0, L)
    offs_p = tl.arange(0, NB)
    offs_r = tl.arange(0, RP)
    qbase = Q + bcur * stride_qb + h * stride_qh
    q_lat = tl.load(qbase + offs_l).to(tl.float32)
    q_rope = tl.load(qbase + L + offs_r).to(tl.float32)
    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([L], dtype=tl.float32)
    for start in range(0, seq_len, BLOCK_N):
        offs_n = start + tl.arange(0, BLOCK_N)
        mask = offs_n < seq_len
        phys = tl.load(BlockTable + bcur * stride_btb + offs_n // PAGE,
                       mask=mask, other=0)
        kv_loc = phys * PAGE + offs_n % PAGE                       # [BN]
        base = kv_loc[:, None] * REC                              # [BN,1]
        b = tl.load(Cache + base + offs_p[None, :],
                    mask=mask[:, None], other=0).to(tl.uint32)    # [BN,NB]
        sc = tl.load((Cache + kv_loc * REC + SCALE_OFF).to(tl.pointer_type(tl.float16)),
                     mask=mask, other=0.0).to(tl.float32)          # [BN]
        zp = tl.load((Cache + kv_loc * REC + ZP_OFF).to(tl.pointer_type(tl.float16)),
                     mask=mask, other=0.0).to(tl.float32)
        lat_lo = (b & 0xF).to(tl.float32) * sc[:, None] + zp[:, None]
        lat_hi = ((b >> 4) & 0xF).to(tl.float32) * sc[:, None] + zp[:, None]
        lat = tl.interleave(lat_lo, lat_hi)                        # [BN,L]
        rp_ptr = (Cache + kv_loc[:, None] * REC + ROPE_OFF).to(
            tl.pointer_type(tl.float16)) + offs_r[None, :]
        rp = tl.load(rp_ptr, mask=mask[:, None], other=0.0).to(tl.float32)  # [BN,RP]
        qk = (tl.sum(lat * q_lat[None, :], axis=1)
              + tl.sum(rp * q_rope[None, :], axis=1)) * sm_scale   # [BN]
        qk = tl.where(mask, qk, -float("inf"))
        chunk_max = tl.max(qk, axis=0)
        new_max = tl.maximum(e_max, chunk_max)
        p = tl.exp(qk - new_max)                                   # [BN]
        alpha = tl.exp(e_max - new_max)
        e_sum = e_sum * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * lat, axis=0)       # [L]
        e_max = new_max
    tl.store(O + bcur * stride_ob + h * stride_oh + offs_l, (acc / e_sum).to(O.dtype.element_ty))
    tl.store(Lse + bcur * stride_lb + h, e_max + tl.log(e_sum))


@triton.jit
def _kvarn_mla_gather_dequant_kernel(
    Cache, BlockTable, TokenToSeq, CuSeqLens, SeqStarts,
    B2S, PoolLat, PoolRope, Dst,
    stride_bt, stride_dst,
    stride_pl_b, stride_pl_t, stride_pr_b, stride_pr_t,
    L: tl.constexpr, RP: tl.constexpr, G: tl.constexpr, REC: tl.constexpr,
    SC_OFF: tl.constexpr, ZP_OFF: tl.constexpr, SR_OFF: tl.constexpr,
    RP_OFF: tl.constexpr, NUM_BLOCKS_LOOKUP: tl.constexpr,
):
    """Context gather for KVarN-MLA prefill (chunked prefill and prefix-cache
    hits): grid (num_tokens,). For each cached context token, read the ROTATED
    fp16 latent + rope from the tail pool (unflushed block, slot >= 0 in the
    block->slot lookup) or dequant it from the per-block int4 TILE record
    (flushed block: token-major packed latent, per-channel scale/zp, per-token
    s_row — same layout the tile decode kernel reads), and write
    ``[lat_rot, rope]`` to the prefill workspace row. The caller un-rotates the
    latent (one batched matmul by H^T) to recover ``kv_c_normed``. Replaces the
    old gather, which read the tile cache with the legacy PER-TOKEN record
    layout — wrong format for the tile method — and knew nothing of the pool."""
    t = tl.program_id(0)
    seq = tl.load(TokenToSeq + t)
    pos = t - tl.load(CuSeqLens + seq) + tl.load(SeqStarts + seq)
    phys = tl.load(BlockTable + seq * stride_bt + pos // G)
    intra = (pos % G).to(tl.int64)
    offs_l = tl.arange(0, L)
    offs_r = tl.arange(0, RP)
    dbase = Dst + t * stride_dst
    in_range = (phys >= 0) & (phys < NUM_BLOCKS_LOOKUP)
    slot = tl.load(B2S + phys, mask=in_range, other=-1)
    if slot >= 0:
        lat = tl.load(PoolLat + slot.to(tl.int64) * stride_pl_b
                      + intra * stride_pl_t + offs_l).to(tl.float32)
        rope = tl.load(PoolRope + slot.to(tl.int64) * stride_pr_b
                       + intra * stride_pr_t + offs_r).to(tl.float32)
        tl.store(dbase + offs_l, lat.to(Dst.dtype.element_ty))
        tl.store(dbase + L + offs_r, rope.to(Dst.dtype.element_ty))
    else:
        base = phys.to(tl.int64) * REC
        pk = tl.load(Cache + base + intra * (L // 2)
                     + tl.arange(0, L // 2)).to(tl.uint32)
        sc = tl.load((Cache + base + SC_OFF).to(tl.pointer_type(tl.float16))
                     + offs_l).to(tl.float32)
        zp = tl.load((Cache + base + ZP_OFF).to(tl.pointer_type(tl.float16))
                     + offs_l).to(tl.float32)
        pt = tl.load((Cache + base + SR_OFF).to(tl.pointer_type(tl.float16))
                     + intra).to(tl.float32)
        lat = tl.interleave((pk & 0xF).to(tl.float32),
                            ((pk >> 4) & 0xF).to(tl.float32))
        lat = (lat * sc + zp) * pt
        rope = tl.load((Cache + base + RP_OFF).to(tl.pointer_type(tl.float16))
                       + intra * RP + offs_r).to(tl.float32)
        tl.store(dbase + offs_l, lat.to(Dst.dtype.element_ty))
        tl.store(dbase + L + offs_r, rope.to(Dst.dtype.element_ty))


@triton.jit
def _kvarn_mla_scatter_store_kernel(
    Lat_in_ptr,          # [T, R]      fp16 incoming latent (kv_c_normed)
    Rope_in_ptr,         # [T, ROPE]   fp16 incoming rope (k_pe)
    Slot_mapping_ptr,    # [T]         int64  (slot < 0 => pad/skip)
    Block_to_slot_ptr,   # [num_blocks_lookup] int32 (-1 = no pool slot)
    Pool_lat_ptr,        # [POOL, GROUP, R]     fp16
    Pool_rope_ptr,       # [POOL, GROUP, ROPE]  fp16
    stride_lat_t, stride_rope_t,
    stride_pl_b, stride_pl_t, stride_pr_b, stride_pr_t,
    R: tl.constexpr, ROPE: tl.constexpr, GROUP: tl.constexpr,
    NUM_BLOCKS_LOOKUP: tl.constexpr,
):
    """Graph-safe tensorized store: scatter one incoming token row (latent+rope)
    into the sparse fp16 tail pool at slot=Block_to_slot[slot_mapping[i]//GROUP],
    pos=slot_mapping[i]%GROUP. No Python loop / host-sync -> capturable. Tokens
    whose block has no pool slot (slot<0, i.e. already flushed to int4) or pad
    rows (slot_mapping<0) are skipped. Grid: (T,). Validated: scripts_kvarn_mla/
    kvarn_mla_scatter_store.py."""
    i = tl.program_id(0)
    sm = tl.load(Slot_mapping_ptr + i)
    if sm < 0:
        return
    block_id = sm // GROUP
    pos = (sm % GROUP).to(tl.int64)
    in_range = (block_id >= 0) & (block_id < NUM_BLOCKS_LOOKUP)
    if not in_range:
        return
    slot = tl.load(Block_to_slot_ptr + block_id)
    if slot < 0:
        return
    offs_r = tl.arange(0, R)
    lat = tl.load(Lat_in_ptr + i * stride_lat_t + offs_r)
    tl.store(Pool_lat_ptr + slot.to(tl.int64) * stride_pl_b + pos * stride_pl_t + offs_r, lat)
    offs_rope = tl.arange(0, ROPE)
    rope = tl.load(Rope_in_ptr + i * stride_rope_t + offs_rope)
    tl.store(Pool_rope_ptr + slot.to(tl.int64) * stride_pr_b + pos * stride_pr_t + offs_rope, rope)


@triton.jit
def _kvarn_mla_tile_decode_kernel(
    Q_ptr,             # [B, H, R+ROPE]  (q_lat ALREADY rotated | q_rope)
    Cache_ptr,         # [num_blocks * REC] uint8  int4 tile records
    PoolLat_ptr,       # [POOL, G, R] fp16  (rotated)
    PoolRope_ptr,      # [POOL, G, ROPE] fp16
    BlockTable_ptr,    # [B, max_blocks] int32
    Seqlens_ptr,       # [B] int32
    Block2Slot_ptr,    # [num_blocks] int32  (-1 = flushed/int4)
    O_ptr,             # [B, H, R] fp32  (rotated output)
    Lse_ptr,           # [B, H] fp32
    sm_scale,
    stride_qb, stride_qh, stride_btb, stride_ob, stride_oh, stride_lb,
    stride_plb, stride_plt, stride_prb, stride_prt,
    R: tl.constexpr, ROPE: tl.constexpr, G: tl.constexpr,
    NB: tl.constexpr, SC: tl.constexpr, ZP: tl.constexpr, SR: tl.constexpr,
    RP: tl.constexpr, REC: tl.constexpr, BLOCK_N: tl.constexpr,
    NUM_BLOCKS_LOOKUP: tl.constexpr,
):
    """Fused dual-source KVarN-MLA decode (one (seq, q-head) per program). Reads
    each block from int4 tile cache (dequant in-register) or the rotated fp16 pool
    (in-progress block), online softmax in the ROTATED frame. Validated cos
    0.99999 (scripts_kvarn_mla/kvarn_mla_tile_decode_kernel.py). Output rotated ->
    caller un-rotates by Hᵀ."""
    b = tl.program_id(0)
    h = tl.program_id(1)
    seq_len = tl.load(Seqlens_ptr + b)
    offs_r = tl.arange(0, R)
    offs_rope = tl.arange(0, ROPE)
    half = tl.arange(0, R // 2)
    qbase = Q_ptr + b * stride_qb + h * stride_qh
    q_lat = tl.load(qbase + offs_r).to(tl.float32)
    q_rope = tl.load(qbase + R + offs_rope).to(tl.float32)
    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([R], dtype=tl.float32)
    n_blocks = (seq_len + G - 1) // G
    for j in range(0, n_blocks):
        block_id = tl.load(BlockTable_ptr + b * stride_btb + j)
        slot = tl.load(Block2Slot_ptr + block_id,
                       mask=(block_id >= 0) & (block_id < NUM_BLOCKS_LOOKUP), other=-1)
        base = block_id.to(tl.int64) * REC
        sc = tl.load((Cache_ptr + base + SC).to(tl.pointer_type(tl.float16)) + offs_r,
                     mask=slot < 0, other=0.0).to(tl.float32)
        zp = tl.load((Cache_ptr + base + ZP).to(tl.pointer_type(tl.float16)) + offs_r,
                     mask=slot < 0, other=0.0).to(tl.float32)
        for c0 in range(0, G, BLOCK_N):
            offs_n = c0 + tl.arange(0, BLOCK_N)
            tok_mask = offs_n < (seq_len - j * G)
            if slot >= 0:
                pl = PoolLat_ptr + slot.to(tl.int64) * stride_plb + offs_n[:, None] * stride_plt + offs_r[None, :]
                deq = tl.load(pl, mask=tok_mask[:, None], other=0.0).to(tl.float32)
                pr = PoolRope_ptr + slot.to(tl.int64) * stride_prb + offs_n[:, None] * stride_prt + offs_rope[None, :]
                rope = tl.load(pr, mask=tok_mask[:, None], other=0.0).to(tl.float32)
            else:
                pk_ptr = Cache_ptr + base + offs_n[:, None] * (R // 2) + half[None, :]
                pk = tl.load(pk_ptr, mask=tok_mask[:, None], other=0).to(tl.uint32)
                lo = (pk & 0xF).to(tl.float32)
                hi = ((pk >> 4) & 0xF).to(tl.float32)
                q4 = tl.interleave(lo, hi)
                srow = tl.load((Cache_ptr + base + SR).to(tl.pointer_type(tl.float16)) + offs_n,
                               mask=tok_mask, other=0.0).to(tl.float32)
                deq = (q4 * sc[None, :] + zp[None, :]) * srow[:, None]
                rp_ptr = (Cache_ptr + base + RP).to(tl.pointer_type(tl.float16)) + offs_n[:, None] * ROPE + offs_rope[None, :]
                rope = tl.load(rp_ptr, mask=tok_mask[:, None], other=0.0).to(tl.float32)
            score = (tl.sum(deq * q_lat[None, :], axis=1)
                     + tl.sum(rope * q_rope[None, :], axis=1)) * sm_scale
            score = tl.where(tok_mask, score, -float("inf"))
            chunk_max = tl.max(score, axis=0)
            new_max = tl.maximum(e_max, chunk_max)
            p = tl.exp(score - new_max)
            alpha = tl.exp(e_max - new_max)
            e_sum = e_sum * alpha + tl.sum(p, axis=0)
            acc = acc * alpha + tl.sum(p[:, None] * deq, axis=0)
            e_max = new_max
    tl.store(O_ptr + b * stride_ob + h * stride_oh + offs_r, acc / e_sum)
    tl.store(Lse_ptr + b * stride_lb + h, e_max + tl.log(e_sum))


@triton.jit
def _kvarn_mla_splitk_stage1(
    Q_ptr, Cache_ptr, PoolLat_ptr, PoolRope_ptr, BlockTable_ptr, Seqlens_ptr,
    Block2Slot_ptr, PartO_ptr, PartLse_ptr, sm_scale,
    stride_qb, stride_qh, stride_btb,
    stride_pob, stride_poh, stride_pos, stride_plseb, stride_plseh,
    stride_plb, stride_plt, stride_prb, stride_prt,
    R: tl.constexpr, ROPE: tl.constexpr, G: tl.constexpr,
    NB: tl.constexpr, SC: tl.constexpr, ZP: tl.constexpr, SR: tl.constexpr,
    RP: tl.constexpr, REC: tl.constexpr, BLOCK_N: tl.constexpr,
    NUM_SPLITS: tl.constexpr, NUM_BLOCKS_LOOKUP: tl.constexpr,
):
    """Split-K stage1: partial attention over this split's KV-block slice.
    Stores normalized partial o_s [R] + lse_s. Validated cos 1.0
    (scripts_kvarn_mla/kvarn_mla_splitk_decode.py)."""
    b = tl.program_id(0); h = tl.program_id(1); s = tl.program_id(2)
    seq_len = tl.load(Seqlens_ptr + b)
    n_blocks = (seq_len + G - 1) // G
    bps = (n_blocks + NUM_SPLITS - 1) // NUM_SPLITS
    blk0 = s * bps
    blk1 = tl.minimum(blk0 + bps, n_blocks)
    offs_r = tl.arange(0, R); offs_rope = tl.arange(0, ROPE); half = tl.arange(0, R // 2)
    qbase = Q_ptr + b * stride_qb + h * stride_qh
    q_lat = tl.load(qbase + offs_r).to(tl.float32)
    q_rope = tl.load(qbase + R + offs_rope).to(tl.float32)
    e_max = -float("inf"); e_sum = 0.0
    acc = tl.zeros([R], dtype=tl.float32)
    for j in range(blk0, blk1):
        block_id = tl.load(BlockTable_ptr + b * stride_btb + j)
        slot = tl.load(Block2Slot_ptr + block_id,
                       mask=(block_id >= 0) & (block_id < NUM_BLOCKS_LOOKUP), other=-1)
        base = block_id.to(tl.int64) * REC
        sc = tl.load((Cache_ptr + base + SC).to(tl.pointer_type(tl.float16)) + offs_r,
                     mask=slot < 0, other=0.0).to(tl.float32)
        zp = tl.load((Cache_ptr + base + ZP).to(tl.pointer_type(tl.float16)) + offs_r,
                     mask=slot < 0, other=0.0).to(tl.float32)
        for c0 in range(0, G, BLOCK_N):
            offs_n = c0 + tl.arange(0, BLOCK_N)
            tok_mask = offs_n < (seq_len - j * G)
            if slot >= 0:
                pl = PoolLat_ptr + slot.to(tl.int64) * stride_plb + offs_n[:, None] * stride_plt + offs_r[None, :]
                deq = tl.load(pl, mask=tok_mask[:, None], other=0.0).to(tl.float32)
                pr = PoolRope_ptr + slot.to(tl.int64) * stride_prb + offs_n[:, None] * stride_prt + offs_rope[None, :]
                rope = tl.load(pr, mask=tok_mask[:, None], other=0.0).to(tl.float32)
            else:
                pk = tl.load(Cache_ptr + base + offs_n[:, None] * (R // 2) + half[None, :],
                             mask=tok_mask[:, None], other=0).to(tl.uint32)
                q4 = tl.interleave((pk & 0xF).to(tl.float32), ((pk >> 4) & 0xF).to(tl.float32))
                srow = tl.load((Cache_ptr + base + SR).to(tl.pointer_type(tl.float16)) + offs_n,
                               mask=tok_mask, other=0.0).to(tl.float32)
                deq = (q4 * sc[None, :] + zp[None, :]) * srow[:, None]
                rp_ptr = (Cache_ptr + base + RP).to(tl.pointer_type(tl.float16)) + offs_n[:, None] * ROPE + offs_rope[None, :]
                rope = tl.load(rp_ptr, mask=tok_mask[:, None], other=0.0).to(tl.float32)
            score = (tl.sum(deq * q_lat[None, :], axis=1) + tl.sum(rope * q_rope[None, :], axis=1)) * sm_scale
            score = tl.where(tok_mask, score, -float("inf"))
            new_max = tl.maximum(e_max, tl.max(score, axis=0))
            p = tl.exp(score - new_max); alpha = tl.exp(e_max - new_max)
            e_sum = e_sum * alpha + tl.sum(p, axis=0)
            acc = acc * alpha + tl.sum(p[:, None] * deq, axis=0)
            e_max = new_max
    empty = blk0 >= n_blocks
    o_s = tl.where(empty, 0.0, acc / tl.where(e_sum > 0, e_sum, 1.0))
    lse_s = tl.where(empty | (e_sum <= 0), -float("inf"), e_max + tl.log(e_sum))
    tl.store(PartO_ptr + b * stride_pob + h * stride_poh + s * stride_pos + offs_r, o_s)
    tl.store(PartLse_ptr + b * stride_plseb + h * stride_plseh + s, lse_s)


@triton.jit
def _kvarn_mla_splitk_stage2(
    PartO_ptr, PartLse_ptr, O_ptr, Lse_ptr,
    stride_pob, stride_poh, stride_pos, stride_plseb, stride_plseh,
    stride_ob, stride_oh, stride_lb,
    R: tl.constexpr, NUM_SPLITS: tl.constexpr,
):
    """Split-K stage2: LSE-combine the NUM_SPLITS partials -> final O[R], Lse."""
    b = tl.program_id(0); h = tl.program_id(1)
    offs_r = tl.arange(0, R); s_off = tl.arange(0, NUM_SPLITS)
    lse = tl.load(PartLse_ptr + b * stride_plseb + h * stride_plseh + s_off)
    gm = tl.max(lse, axis=0)
    w = tl.exp(lse - gm); wsum = tl.sum(w, axis=0); w = w / wsum
    acc = tl.zeros([R], dtype=tl.float32)
    for s in range(0, NUM_SPLITS):
        o_s = tl.load(PartO_ptr + b * stride_pob + h * stride_poh + s * stride_pos + offs_r)
        acc += o_s * tl.sum(tl.where(s_off == s, w, 0.0), axis=0)
    tl.store(O_ptr + b * stride_ob + h * stride_oh + offs_r, acc)
    tl.store(Lse_ptr + b * stride_lb + h, gm + tl.log(wsum))


@triton.jit
def _kvarn_mla_grouped_stage1(
    Q_ptr, Cache_ptr, PoolLat_ptr, PoolRope_ptr, BlockTable_ptr, Seqlens_ptr,
    Block2Slot_ptr, PartO_ptr, PartLse_ptr, sm_scale,
    stride_qb, stride_qh, stride_btb,
    stride_pob, stride_poh, stride_pos, stride_plseb, stride_plseh,
    stride_plb, stride_plt, stride_prb, stride_prt,
    H: tl.constexpr, HGROUP: tl.constexpr,
    R: tl.constexpr, ROPE: tl.constexpr, G: tl.constexpr,
    NB: tl.constexpr, SC: tl.constexpr, ZP: tl.constexpr, SR: tl.constexpr,
    RP: tl.constexpr, REC: tl.constexpr, BLOCK_N: tl.constexpr,
    NUM_SPLITS: tl.constexpr, NUM_BLOCKS_LOOKUP: tl.constexpr,
):
    """GROUPED-HEAD split-K stage1 (the fast kernel). All HGROUP query heads share
    the dequant of each block (MLA latent is shared) -> H-fold less dequant; scores
    and V-accumulate via tl.dot (tensor cores). fp16 dot operands (fp32 accumulate)
    to fit shared mem. Validated cos 1.0 (kvarn_mla_grouped_decode.py)."""
    b = tl.program_id(0); hg = tl.program_id(1); s = tl.program_id(2)
    seq_len = tl.load(Seqlens_ptr + b)
    n_blocks = (seq_len + G - 1) // G
    bps = (n_blocks + NUM_SPLITS - 1) // NUM_SPLITS
    blk0 = s * bps; blk1 = tl.minimum(blk0 + bps, n_blocks)
    offs_h = hg * HGROUP + tl.arange(0, HGROUP); hmask = offs_h < H
    offs_r = tl.arange(0, R); offs_rope = tl.arange(0, ROPE); half = tl.arange(0, R // 2)
    offs_n = tl.arange(0, BLOCK_N)
    qrow = Q_ptr + b * stride_qb + offs_h[:, None] * stride_qh
    q_lat = tl.load(qrow + offs_r[None, :], mask=hmask[:, None], other=0.0).to(tl.float16)
    q_rope = tl.load(qrow + R + offs_rope[None, :], mask=hmask[:, None], other=0.0).to(tl.float16)
    m_i = tl.full([HGROUP], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([HGROUP], dtype=tl.float32)
    acc = tl.zeros([HGROUP, R], dtype=tl.float32)
    for j in range(blk0, blk1):
        block_id = tl.load(BlockTable_ptr + b * stride_btb + j)
        slot = tl.load(Block2Slot_ptr + block_id,
                       mask=(block_id >= 0) & (block_id < NUM_BLOCKS_LOOKUP), other=-1)
        base = block_id.to(tl.int64) * REC
        sc = tl.load((Cache_ptr + base + SC).to(tl.pointer_type(tl.float16)) + offs_r,
                     mask=slot < 0, other=0.0).to(tl.float32)
        zp = tl.load((Cache_ptr + base + ZP).to(tl.pointer_type(tl.float16)) + offs_r,
                     mask=slot < 0, other=0.0).to(tl.float32)
        for c0 in range(0, G, BLOCK_N):
            nn = c0 + offs_n
            tok_mask = nn < (seq_len - j * G)
            if slot >= 0:
                pl = PoolLat_ptr + slot.to(tl.int64) * stride_plb + nn[:, None] * stride_plt + offs_r[None, :]
                deq = tl.load(pl, mask=tok_mask[:, None], other=0.0).to(tl.float32)
                pr = PoolRope_ptr + slot.to(tl.int64) * stride_prb + nn[:, None] * stride_prt + offs_rope[None, :]
                rope = tl.load(pr, mask=tok_mask[:, None], other=0.0).to(tl.float32)
            else:
                pk = tl.load(Cache_ptr + base + nn[:, None] * (R // 2) + half[None, :],
                             mask=tok_mask[:, None], other=0).to(tl.uint32)
                q4 = tl.interleave((pk & 0xF).to(tl.float32), ((pk >> 4) & 0xF).to(tl.float32))
                srow = tl.load((Cache_ptr + base + SR).to(tl.pointer_type(tl.float16)) + nn,
                               mask=tok_mask, other=0.0).to(tl.float32)
                deq = (q4 * sc[None, :] + zp[None, :]) * srow[:, None]
                rp_ptr = (Cache_ptr + base + RP).to(tl.pointer_type(tl.float16)) + nn[:, None] * ROPE + offs_rope[None, :]
                rope = tl.load(rp_ptr, mask=tok_mask[:, None], other=0.0).to(tl.float32)
            deqh = deq.to(tl.float16)
            scores = (tl.dot(q_lat, tl.trans(deqh))
                      + tl.dot(q_rope, tl.trans(rope.to(tl.float16)))) * sm_scale
            scores = tl.where(tok_mask[None, :], scores, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            p = tl.exp(scores - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), deqh)
            m_i = m_new
    empty = blk0 >= n_blocks
    o = tl.where(empty | (l_i <= 0)[:, None], 0.0, acc / tl.where(l_i > 0, l_i, 1.0)[:, None])
    lse = tl.where(empty | (l_i <= 0), -float("inf"), m_i + tl.log(l_i))
    po = PartO_ptr + b * stride_pob + offs_h[:, None] * stride_poh + s * stride_pos + offs_r[None, :]
    tl.store(po, o, mask=hmask[:, None])
    tl.store(PartLse_ptr + b * stride_plseb + offs_h * stride_plseh + s, lse, mask=hmask)


from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd

logger = init_logger(__name__)


class TritonMLAMetadataBuilder(MLACommonMetadataBuilder[MLACommonMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        md = super().build(common_prefix_len, common_attn_metadata, fast_build)
        impls = getattr(TritonMLAImpl, "_kvarn_impls", [])
        if not impls or not impls[0]._kvarn_graph:
            return md
        # ── KVarN-MLA: eager Sinkhorn tile-flush + sparse-pool slot management,
        # run here (outside any captured region) between graph replays. The
        # captured do_kv_cache_update only SCATTERS into the pool; this builder
        # FLUSHES filled 128-token tiles into the int4 cache and (re)allocates
        # pool slots. Mirrors kvarn_attn.py build() (no sink: MLA flushes all
        # full blocks). Flush is keyed off PREV step's seq_len (= pool fill now),
        # because this step's token is written AFTER the builder, in forward.
        cls = TritonMLAImpl
        G = impls[0]._kvarn_group
        # seq_lens from the CPU copy vLLM already maintains -> NO D2H sync.
        slc = common_attn_metadata.seq_lens_cpu_upper_bound
        if slc is None:
            slc = common_attn_metadata.seq_lens.cpu()
        seq_lens_cpu = slc.tolist()
        B = len(seq_lens_cpu)
        n_tok = common_attn_metadata.num_actual_tokens

        # ── FAST-SKIP steady-state decode steps ─────────────────────────────
        # Profiling showed GPU util drops to ~66% at long context: the builder's
        # per-step block_table D2H + Python slot bookkeeping doesn't overlap the
        # graph replay -> ~34% GPU idle. On a pure-decode step that crosses NO
        # block boundary, no tile flush and no slot alloc happen (the tail slot is
        # unchanged), so the entire builder is a no-op. Boundary-adjacent steps
        # (sl%G in {0,1}: 0 => a new tail block is allocated this step; 1 => the
        # previous step filled a block -> flush) still run the full path below.
        # Detected purely from seq_lens_cpu (no D2H).
        prev = cls.__dict__.get("_kvarn_prev_seqlens")
        if (not os.environ.get("KVARN_MLA_NOSKIP")
                and n_tok == B and prev is not None and len(prev) == B
                and all(seq_lens_cpu[i] == prev[i] + 1 and seq_lens_cpu[i] % G > 1
                        for i in range(B))):
            cls._kvarn_prev_seqlens = seq_lens_cpu
            return md
        cls._kvarn_prev_seqlens = seq_lens_cpu

        bt = common_attn_metadata.block_table_tensor
        device = bt.device
        block_table_cpu = bt.tolist()                 # 1 D2H (block ids for slot mgmt)

        # Pool sizing: one in-flight tail (+ one-step flush latency) per request,
        # plus every block a single chunked-prefill step can write — a chunk
        # spans up to max_num_batched_tokens / G blocks across the batch, and
        # those blocks hold pool slots until flushed on the following step.
        sched = self.vllm_config.scheduler_config
        pool_slots = int(os.environ.get("KVARN_MLA_POOL_SLOTS", "0")) or max(
            2 * sched.max_num_seqs
            + (sched.max_num_batched_tokens + G - 1) // G, 64)
        # b2s_t MUST be sized to the FULL block-id range up front: its pointer is
        # baked into the captured CUDA graph (forward_mqa), so a runtime realloc
        # (triggered when a block id first exceeds the current size) leaves the
        # graph reading a stale frozen tensor -> KV corruption that scales with
        # batch/context (only manifests once block ids exceed the initial 1024).
        # The builder runs during warmup (before capture) and HAS vllm_config
        # (the impl does not), so seed nb_hint with num_gpu_blocks here.
        nb_hint = (getattr(self.vllm_config.cache_config, "num_gpu_blocks", None)
                   or 0)
        if nb_hint < 1024:
            nb_hint = 1024
        for row in block_table_cpu:
            for b in row:
                if b >= 0 and b + 1 > nb_hint:
                    nb_hint = b + 1
        for impl in impls:
            impl._ensure_kvarn_pool(device, nb_hint, pool_slots)
        b2s_t = cls._kvarn_b2slot_t[device]
        b2s_dict = cls._kvarn_b2slot_dict
        free = cls._kvarn_free_slots
        fill = cls._kvarn_fill

        # ── Sharing-safe slot lifecycle (prefix caching + chunked prefill) ──
        # No per-request state: vLLM's prefix caching shares physical blocks
        # across LIVE requests and recycles ids across finished ones, so any
        # request-identity proxy (the old prev_sl/watermark keyed by row[0])
        # collides under sharing — two requests with a common prefix share
        # row[0], their flush state flip-flops, and blocks get quantized
        # half-written or their slots freed in use (the issue #10 "illegal
        # memory access after ~37 min" / repetition-collapse class). Everything
        # below derives from per-step facts instead:
        #   committed = seq_len - query_len   (tokens written BEFORE this step;
        #     this step's tokens land in forward, after the builder — also
        #     correct under chunked prefill and spec decode, where query_len
        #     covers the whole chunk / verify window)
        #   b2s_dict membership = "block is unflushed" (ground truth: a flush
        #     frees the slot, so a slot-holding block below the committed
        #     boundary is exactly a full-but-unflushed block).
        qsl = getattr(common_attn_metadata, "query_start_loc_cpu", None)
        if qsl is not None:
            qsl = qsl.tolist()
            query_lens = [qsl[i + 1] - qsl[i] for i in range(B)]
        else:
            query_lens = [(n_tok // B if B else 1)] * B

        blocks_needed: set[int] = set()
        flush_seen: set[int] = set()
        flush_q: list[int] = []
        for b in range(B):
            row = block_table_cpu[b]
            sl = seq_lens_cpu[b]
            if not row or sl <= 0:
                continue
            committed = max(sl - query_lens[b], 0)
            # Blocks receiving writes this step need pool slots. Record how
            # full each will be AFTER the step: if its owner finishes on the
            # step that fills it, the reclaim below must flush (not discard).
            # For a cache-hit prefill, committed = the cached length, so the
            # shared context blocks correctly need no slots here.
            for k in range(committed // G, min((sl - 1) // G, len(row) - 1) + 1):
                bid = row[k]
                if bid >= 0:
                    blocks_needed.add(bid)
                    fill[bid] = min(sl, (k + 1) * G) - k * G
            # FLUSH detection: walk backward from the committed boundary while
            # blocks still hold pool slots — those are full-but-unflushed.
            # Stops at the first slotless block (flushes happen in order, so
            # everything earlier is already int4). Idempotent under sharing:
            # a co-owner finds the block already queued (or slotless) and stops.
            k = committed // G - 1
            while 0 <= k < len(row):
                bid = row[k]
                if bid < 0 or bid in flush_seen or bid not in b2s_dict:
                    break
                flush_seen.add(bid)
                flush_q.append(bid)
                k -= 1

        # RECLAIM: slot-holding blocks neither written this step nor already
        # queued belong to finished (or descheduled) requests. A COMPLETE one
        # is flushed — vLLM's prefix cache may hand the block to a future
        # request, which must find a valid int4 tile (the old discard left
        # stale tile bytes for the cache hit to read). A partial one is safe
        # to discard: vLLM never prefix-caches partial blocks.
        discard_ids: list[int] = []
        for bid in list(b2s_dict):
            if bid in blocks_needed or bid in flush_seen:
                continue
            if fill.get(bid, 0) >= G:
                flush_seen.add(bid)
                flush_q.append(bid)
            else:
                discard_ids.append(bid)

        if flush_q:
            # Batch ALL (layer, block) flush tiles into one vectorized flush
            # instead of per-tile eager ops (the 25%-GPU / 66%-util hot spot).
            iters = int(os.environ.get("KVARN_SINKHORN_ITERS", "16"))
            flush_list = []
            for impl in impls:
                if impl._kv_cache_ref is None:
                    continue
                for bid in flush_q:
                    slot = b2s_dict.get(bid)
                    if slot is not None:
                        flush_list.append((impl, bid, slot))
            if os.environ.get("KVARN_MLA_PERTILE"):
                for impl, bid, slot in flush_list:      # isolation: per-tile flush
                    impl._kvarn_flush_tile(impl._kvarn_lat_pool[slot],
                                           impl._kvarn_rope_pool[slot],
                                           impl._kv_cache_ref, bid, already_rotated=True)
            else:
                cls._kvarn_batched_flush(flush_list, iters)
        for bid in flush_q:
            slot = b2s_dict.pop(bid, None)
            fill.pop(bid, None)
            if slot is not None:
                free.append(slot)
                if bid < b2s_t.shape[0]:
                    b2s_t[bid] = -1
        for bid in discard_ids:
            slot = b2s_dict.pop(bid)
            fill.pop(bid, None)
            free.append(slot)
            if bid < b2s_t.shape[0]:
                b2s_t[bid] = -1

        # (3) ALLOCATE slots for needed blocks lacking one.
        for bid in blocks_needed:
            if bid in b2s_dict:
                continue
            if not free:
                raise RuntimeError(
                    f"KVarN-MLA pool exhausted (size={pool_slots}); raise "
                    f"KVARN_MLA_POOL_SLOTS")
            slot = free.pop()
            b2s_dict[bid] = slot
            if bid < b2s_t.shape[0]:
                b2s_t[bid] = slot
            if not os.environ.get("KVARN_MLA_NOZERO"):
                # zero the (possibly stale, just-reused) slot in every layer's pool
                # so the decode never reads a previous block's leftover tokens.
                for impl in impls:
                    if impl._kvarn_lat_pool is not None:
                        impl._kvarn_lat_pool[slot].zero_()
                        impl._kvarn_rope_pool[slot].zero_()
        if os.environ.get("KVARN_MLA_DBG"):
            vals = list(b2s_dict.values())
            if len(vals) != len(set(vals)):
                from collections import Counter
                dup = [s for s, c in Counter(vals).items() if c > 1]
                print(f"[KVDBG] SLOT COLLISION dup={dup} b2s={dict(b2s_dict)} "
                      f"seqlens={seq_lens_cpu}", flush=True)
            miss = [b for b in blocks_needed if b not in b2s_dict]
            if miss:
                print(f"[KVDBG] MISSING {miss} seqlens={seq_lens_cpu}", flush=True)
            # also flag block_ids shared across sequences this step (prefix-cache-like)
            firstcols = [block_table_cpu[b][0] for b in range(B)
                         if block_table_cpu[b] and block_table_cpu[b][0] >= 0]
            if len(firstcols) != len(set(firstcols)):
                print(f"[KVDBG] SHARED row0 across seqs: {firstcols}", flush=True)
        return md


class TritonMLABackend(MLACommonBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "kvarn_mla_k4_g128",
        "kvarn_k4v2_g128",  # alias: on an MLA model the dense KVarN dtype routes
                            # to the MLA latent-quant path (k4 latent), so users can
                            # pass the same --kv-cache-dtype on dense and MLA models.
    ]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size % 16 == 0

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA"

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return True

    @staticmethod
    def get_impl_cls() -> type["TritonMLAImpl"]:
        return TritonMLAImpl

    @staticmethod
    def get_builder_cls() -> type["TritonMLAMetadataBuilder"]:
        return TritonMLAMetadataBuilder

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True


class TritonMLAImpl(MLACommonImpl[MLACommonMetadata]):
    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        **mla_args,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **mla_args,
        )

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "TritonMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "TritonMLAImpl"
            )

        # For FP8 KV cache, we dequantize to BF16 on load inside the
        # Triton kernel. Tell the common layer not to quantize queries
        # to FP8 — we handle FP8 KV cache with BF16 queries (Mode 1).
        if is_quantized_kv_cache(self.kv_cache_dtype):
            self.supports_quant_query_input = False

        self._sm_count = current_platform.num_compute_units()
        # ANY kvarn_ dtype on an MLA model -> KVarN latent quant (the dense
        # kvarn_attn backend never runs for MLA, so this impl owns all kvarn_
        # variants). Lets users pass the same --kv-cache-dtype (e.g.
        # kvarn_k4v2_g128) on dense and MLA models with no code change.
        self._is_kvarn_mla = str(self.kv_cache_dtype).startswith("kvarn_")
        # CUDA-graph path: sparse fp16 tail pool + tensorized scatter store
        # (graph-safe) + flush-in-builder + dequant->stock decode. ON by default
        # (validated byte-identical to eager + ~8x faster); KVARN_MLA_GRAPH=0
        # forces the legacy eager dict-staging path.
        self._kvarn_graph = bool(int(os.environ.get("KVARN_MLA_GRAPH", "1")))
        if self._is_kvarn_mla:
            self._kvarn_bits = 4
            self._kvarn_group = 128
            # per-block fp16 staging for unflushed tokens: block_id -> dict(
            #   lat [n,R] fp16, rope [n,ROPE] fp16). Flushed at 128-token fill via
            #   the full KVarN tile recipe (Hadamard+Sinkhorn+per-channel RTN).
            self._kvarn_stage: dict[int, dict] = {}
            self._kvarn_H = None  # lazy orthonormal Hadamard [R,R]
            # ── graph-safe sparse tail pool (per-layer fp16) ─────────────────
            self._kvarn_lat_pool: torch.Tensor | None = None    # [POOL, G, R] fp16
            self._kvarn_rope_pool: torch.Tensor | None = None   # [POOL, G, ROPE] fp16
            self._kv_cache_ref: torch.Tensor | None = None      # this layer's int4 cache
            cls = type(self)
            # Class-level allocator (slot index addresses the SAME logical block
            # in every layer's pool) + impl registry (builder flushes all layers).
            if not hasattr(cls, "_kvarn_b2slot_dict"):
                cls._kvarn_b2slot_dict = {}                     # block_id -> slot
                cls._kvarn_free_slots = None                    # list[int] | None
                cls._kvarn_pool_size = 0
                cls._kvarn_b2slot_t = {}                        # device -> int32[num_blocks]
                cls._kvarn_impls = []                           # list of live impls
                # block_id -> tokens present in the pool for that block after
                # the current step. Keyed by PHYSICAL block (not request), so it
                # stays correct when prefix caching shares blocks across
                # requests; a partial block has exactly one writer, so the
                # value has a single source. Drives flush-on-reclaim.
                cls._kvarn_fill = {}
            if self._kvarn_graph:
                cls._kvarn_impls.append(self)

    def _kvarn_hadamard(self, n: int, device, dtype):
        if self._kvarn_H is None:
            H = torch.ones(1, 1, dtype=torch.float32)
            while H.shape[0] < n:
                H = torch.cat(
                    [torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
            self._kvarn_H = (H / n ** 0.5).to(device=device, dtype=torch.float32)
        return self._kvarn_H

    def _kvarn_hadamard_act(self, n: int, device, dtype):
        """Hadamard cast to the activation dtype (bf16) and cached, so the query/
        output/store rotations run as bf16 TENSOR-CORE GEMMs instead of fp32 SIMT
        sgemm (the profiled hot spot). The latent is 4-bit-quantized so bf16
        rotation is well within noise."""
        h = getattr(self, "_kvarn_Hb", None)
        if h is None or h.dtype != dtype:
            self._kvarn_Hb = self._kvarn_hadamard(n, device, dtype).to(dtype)
        return self._kvarn_Hb

    def _kvarn_flush_tile(self, lat, rope, kv_cache, block_id, already_rotated=False):
        """Full KVarN tile flush: lat [G,R] fp16, rope [G,ROPE] fp16 -> write the
        int4 tile record (Hadamard + Sinkhorn + per-channel RTN) to kv_cache[block].
        already_rotated: lat is already c@H (graph path stores rotated in the pool),
        so skip the rotation matmul."""
        from vllm.model_executor.layers.quantization.kvarn.sinkhorn import (
            variance_normalize,
        )
        G, R = lat.shape
        ROPE = rope.shape[-1]
        bits = self._kvarn_bits
        qmax = (1 << bits) - 1
        NB, SC, ZP, SR, RP, REC, _ = kvarn_mla_tile_layout(R, ROPE, G, bits)
        if already_rotated:
            rot = lat.float().t().contiguous()                # [R, G]; lat = c@H already
        else:
            H = self._kvarn_hadamard(R, lat.device, lat.dtype)
            rot = (lat.float() @ H).t().contiguous()          # [R, G] rotated frame
        bal, s_col, s_row = variance_normalize(rot)           # s_col[1,G] s_row[R,1]
        lo = bal.amin(1, keepdim=True); hi = bal.amax(1, keepdim=True)
        scale = ((hi - lo) / qmax).clamp_min(1e-8)            # [R,1] per-channel
        q = torch.clamp(torch.round((bal - lo) / scale), 0, qmax).to(torch.uint8)
        scale_abs = (scale * s_row).squeeze(1)                # [R] (absorb per-ch sinkhorn)
        zp_abs = (lo * s_row).squeeze(1)                      # [R]
        per_tok = s_col.squeeze(0)                            # [G] per-token sinkhorn
        qT = q.t().contiguous()                               # [G, R] token-major
        packed = (qT[:, 0::2] | (qT[:, 1::2] << 4)).contiguous()
        rec = kv_cache.view(-1, REC)[block_id]
        rec[:NB] = packed.reshape(-1)
        rec[SC:SC + R * 2] = scale_abs.to(torch.float16).view(torch.uint8)
        rec[ZP:ZP + R * 2] = zp_abs.to(torch.float16).view(torch.uint8)
        rec[SR:SR + G * 2] = per_tok.to(torch.float16).view(torch.uint8)
        rec[RP:RP + G * ROPE * 2] = rope.reshape(-1).to(torch.float16).view(torch.uint8)
        import os as _os
        if _os.environ.get("KVARN_MLA_DBG"):
            if not hasattr(self, "_dbg"):
                self._dbg = {}
            self._dbg[block_id] = ((lat.float() @ H).clone(), rope.float().clone())

    @classmethod
    def _kvarn_batched_flush(cls, flush_list, iters):
        """BATCHED full-method flush (the perf fix). flush_list=[(impl,bid,slot)].
        Stacks ALL (layer,block) rotated tiles into one [N,R,G] tensor and runs
        Hadamard(already-applied)+Sinkhorn+per-channel RTN VECTORIZED over the
        batch — replacing ~N separate eager flushes (each ~dozens of tiny
        launch-bound reduce/elementwise kernels + a per-iter host sync). Profiling
        showed the per-tile flush was 25% of GPU time + the cause of 66% util.
        Sinkhorn here is sync-free (torch.where best-tracking, no host `if`) and
        numerically == the per-tile variance_normalize (NOT no-best-track: that
        earlier 'inert' assumption was WRONG — it dropped GLM-4.7-Flash AIME to 20%
        vs 47% per-tile; best-tracking restores it while keeping the batched speed)."""
        if not flush_list:
            return
        i0 = flush_list[0][0]
        G = i0._kvarn_group; R = i0.kv_lora_rank; ROPE = i0.qk_rope_head_dim
        bits = i0._kvarn_bits; qmax = (1 << bits) - 1
        NB, SC, ZP, SR, RP, REC, _ = kvarn_mla_tile_layout(R, ROPE, G, bits)
        # gather rotated latent [N,G,R] -> [N,R,G]; rope [N,G,ROPE]
        lat = torch.stack([im._kvarn_lat_pool[s] for im, _, s in flush_list]).float()
        rope = torch.stack([im._kvarn_rope_pool[s] for im, _, s in flush_list])
        m = lat.transpose(1, 2).contiguous()                          # [N,R,G] (rotated)
        N = m.shape[0]
        # Batched log-domain Sinkhorn WITH best-so-far tracking, vectorized per-tile
        # and SYNC-FREE (torch.where mask, not a host `if`). This matches the
        # single-tile variance_normalize() EXACTLY (same iters/clamps + best-tracking)
        # so the batched flush is numerically == the per-tile flush. The prior
        # no-best-tracking version (took the LAST iter, not the lowest-imbalance one)
        # corrupted accuracy at high batch: GLM-4.7-Flash AIME25 was 20% (batched) vs
        # 47% (per-tile). cols=R-axis (dim1), rows=G-axis (dim2).
        def _imb(t):                                                  # [N,R,G] -> [N]
            sc = t.std(dim=1); sr = t.std(dim=2)
            return (sc.amax(-1) / sc.amin(-1).clamp_min(1e-8)
                    + sr.amax(-1) / sr.amin(-1).clamp_min(1e-8))
        ls_c = torch.zeros(N, 1, G, device=m.device)
        ls_r = torch.zeros(N, R, 1, device=m.device)
        cur = m / ls_c.exp() / ls_r.exp()
        imb_best = _imb(cur)
        sc_best = ls_c.exp().clone(); sr_best = ls_r.exp().clone()
        for _ in range(iters):
            cs = cur.std(dim=1, keepdim=True).clamp(1e-3, 1e3)
            ls_c = (ls_c + cs.log()).clip(-0.3, 10.0)
            cur = m / ls_c.exp() / ls_r.exp()
            rs = cur.std(dim=2, keepdim=True).clamp(1e-3, 1e3)
            ls_r = (ls_r + rs.log()).clip(-0.3, 10.0)
            cur = m / ls_c.exp() / ls_r.exp()
            imb = _imb(cur)
            better = imb <= imb_best                                  # [N]
            mc = better.view(N, 1, 1)
            sc_best = torch.where(mc, ls_c.exp(), sc_best)
            sr_best = torch.where(mc, ls_r.exp(), sr_best)
            imb_best = torch.where(better, imb, imb_best)
        s_col = sc_best; s_row = sr_best                              # [N,1,G],[N,R,1]
        bal = m / s_col / s_row
        lo = bal.amin(2, keepdim=True); hi = bal.amax(2, keepdim=True)
        scale = ((hi - lo) / qmax).clamp_min(1e-8)                    # [N,R,1]
        q = torch.clamp(torch.round((bal - lo) / scale), 0, qmax).to(torch.uint8)
        scale_abs = (scale * s_row).squeeze(2).to(torch.float16)      # [N,R]
        zp_abs = (lo * s_row).squeeze(2).to(torch.float16)            # [N,R]
        per_tok = s_col.squeeze(1).to(torch.float16)                  # [N,G]
        qT = q.transpose(1, 2).contiguous()                          # [N,G,R] token-major
        packed = (qT[:, :, 0::2] | (qT[:, :, 1::2] << 4)).contiguous()  # [N,G,R//2]
        rp16 = rope.to(torch.float16)
        # scatter packed records to each (layer, block) cache slot (indexed writes)
        for i, (im, bid, _) in enumerate(flush_list):
            rec = im._kv_cache_ref.view(-1, REC)[bid]
            rec[:NB] = packed[i].reshape(-1).view(torch.uint8)
            rec[SC:SC + R * 2] = scale_abs[i].view(torch.uint8)
            rec[ZP:ZP + R * 2] = zp_abs[i].view(torch.uint8)
            rec[SR:SR + G * 2] = per_tok[i].view(torch.uint8)
            rec[RP:RP + G * ROPE * 2] = rp16[i].reshape(-1).view(torch.uint8)

    def _ensure_kvarn_pool(self, device, num_blocks, pool_slots=None):
        """Allocate the per-layer sparse fp16 tail pool + class-level allocator +
        GPU block->slot lookup. Called from the metadata builder (outside graph
        capture). Pool holds only in-progress (unflushed) blocks (~max_num_seqs);
        flushed history lives int4 in the cache."""
        if torch.cuda.is_current_stream_capturing():
            return
        cls = type(self)
        R, ROPE, G = self.kv_lora_rank, self.qk_rope_head_dim, self._kvarn_group
        if pool_slots is None:
            pool_slots = int(os.environ.get("KVARN_MLA_POOL_SLOTS", "0")) or 512
        if self._kvarn_lat_pool is None:
            self._kvarn_lat_pool = torch.zeros(
                pool_slots, G, R, dtype=torch.float16, device=device)
            self._kvarn_rope_pool = torch.zeros(
                pool_slots, G, ROPE, dtype=torch.float16, device=device)
            if cls._kvarn_free_slots is None or cls._kvarn_pool_size != pool_slots:
                cls._kvarn_free_slots = list(range(pool_slots - 1, -1, -1))
                cls._kvarn_pool_size = pool_slots
                cls._kvarn_b2slot_dict.clear()
        # b2s_t is read into a Python var in forward_mqa and its POINTER is baked
        # into the captured CUDA graph. If it ever reallocates (grows) at runtime,
        # the graph keeps reading the OLD frozen tensor while the builder updates
        # the NEW one -> stale slot maps -> 2nd-generate KV corruption (graph-only;
        # eager re-reads it each call so was unaffected). Size it ONCE to the FULL
        # cache block count so it is allocated before capture and NEVER grows.
        try:
            total_blocks = self.vllm_config.cache_config.num_gpu_blocks or 0
        except Exception:
            total_blocks = 0
        num_blocks = max(num_blocks, 1024, total_blocks)
        existing = cls._kvarn_b2slot_t.get(device)
        if existing is None or existing.shape[0] < num_blocks:
            if existing is not None and torch.cuda.is_current_stream_capturing():
                # Never realloc mid-capture: would orphan the captured pointer.
                return
            new_t = torch.full((num_blocks,), -1, dtype=torch.int32, device=device)
            if existing is not None:                       # preserve live mappings
                new_t[:existing.shape[0]] = existing
            for bid, slot in cls._kvarn_b2slot_dict.items():
                if bid < num_blocks:
                    new_t[bid] = slot
            cls._kvarn_b2slot_t[device] = new_t
            if os.environ.get("KVARN_MLA_DBG"):
                print(f"[KVDBG] b2s_t (RE)ALLOC -> {num_blocks} blocks "
                      f"(had={None if existing is None else existing.shape[0]}, "
                      f"total={total_blocks}, capturing={torch.cuda.is_current_stream_capturing()})",
                      flush=True)

    def do_kv_cache_update(self, kv_c_normed, k_pe, kv_cache, slot_mapping,
                           kv_cache_dtype, k_scale):
        """KVarN-MLA store (full method): buffer incoming fp16 latent+rope per
        block; when a block fills to `group` tokens, flush it via the KVarN tile
        recipe (Hadamard + Sinkhorn + per-channel RTN) into the int4 tile cache.
        Unflushed (partial/sink) blocks stay fp16 in self._kvarn_stage and are
        read directly by the decode."""
        if not self._is_kvarn_mla:
            return super().do_kv_cache_update(
                kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale)
        if kv_cache.numel() == 0:
            return
        # Cache this layer's int4 cache ref so the metadata builder can flush
        # filled tiles into it (eager, between captured replays).
        self._kv_cache_ref = kv_cache
        if self._kvarn_graph:
            # Graph-safe store: scatter incoming fp16 latent+rope into the sparse
            # tail pool (no Python loop / host-sync). Flush (Sinkhorn pack) of a
            # filled 128-token tile happens in TritonMLAMetadataBuilder.build.
            Tn = kv_c_normed.shape[0]
            if Tn == 0:
                return
            if self._kvarn_lat_pool is None:
                # Fallback ensure (e.g. profile run before the builder allocates).
                self._ensure_kvarn_pool(kv_cache.device, kv_cache.shape[0])
            # Store the latent ALREADY ROTATED (c@H) so the fused decode kernel
            # reads pool + int4 in the same rotated frame (no in-kernel rotation).
            Hb = self._kvarn_hadamard_act(self.kv_lora_rank, kv_c_normed.device, kv_c_normed.dtype)
            lat = (kv_c_normed @ Hb).to(torch.float16)   # bf16 tensor-core rotation
            rope = k_pe.reshape(Tn, -1).to(torch.float16)
            b2s = type(self)._kvarn_b2slot_t[kv_cache.device]
            if os.environ.get("KVARN_MLA_DBG") and not torch.cuda.is_current_stream_capturing():
                sm = slot_mapping.flatten().long()
                valid = sm >= 0
                bids = (sm[valid] // self._kvarn_group)
                inr = bids < b2s.shape[0]
                slots = b2s[bids[inr]]
                dropped = int((slots < 0).sum()) + int((~inr).sum())
                if dropped:
                    print(f"[KVDBG] SCATTER DROPPED {dropped}/{int(valid.sum())} tokens "
                          f"(block has no pool slot) Tn={Tn}", flush=True)
            _kvarn_mla_scatter_store_kernel[(Tn,)](
                lat, rope, slot_mapping.flatten().long(), b2s,
                self._kvarn_lat_pool, self._kvarn_rope_pool,
                lat.stride(0), rope.stride(0),
                self._kvarn_lat_pool.stride(0), self._kvarn_lat_pool.stride(1),
                self._kvarn_rope_pool.stride(0), self._kvarn_rope_pool.stride(1),
                R=self.kv_lora_rank, ROPE=self.qk_rope_head_dim,
                GROUP=self._kvarn_group, NUM_BLOCKS_LOOKUP=b2s.shape[0],
            )
            return
        T = kv_c_normed.shape[0]
        R = kv_c_normed.shape[-1]
        G = self._kvarn_group
        rope = k_pe.reshape(T, -1).to(torch.float16)
        ROPE = rope.shape[-1]
        lat = kv_c_normed.to(torch.float16)
        slots = slot_mapping.flatten().long()
        block_ids = (slots // G).tolist()
        offsets = (slots % G).tolist()
        stage = self._kvarn_stage
        for i in range(T):
            bid = block_ids[i]
            off = offsets[i]
            st = stage.get(bid)
            # offset 0 = a freshly (re)allocated block. vLLM reuses physical
            # block ids across finished requests, so a stale staging entry here
            # would contaminate a new sequence -> always start fresh at offset 0.
            if st is None or off == 0:
                st = {
                    "lat": torch.zeros(G, R, dtype=torch.float16, device=lat.device),
                    "rope": torch.zeros(G, ROPE, dtype=torch.float16, device=lat.device),
                    "filled": torch.zeros(G, dtype=torch.bool, device=lat.device),
                }
                stage[bid] = st
            st["lat"][off] = lat[i]
            st["rope"][off] = rope[i]
            st["filled"][off] = True
            if bool(st["filled"].all()):
                self._kvarn_flush_tile(st["lat"], st["rope"], kv_cache, bid)
                del stage[bid]

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if type(q) is tuple:
            q = torch.cat(q, dim=-1)

        assert isinstance(q, torch.Tensor)
        B = q.shape[0]
        q_num_heads = q.shape[1]
        o = torch.zeros(
            B, q_num_heads, self.kv_lora_rank, dtype=q.dtype, device=q.device
        )
        lse = torch.zeros(B, q_num_heads, dtype=q.dtype, device=q.device)

        if self._is_kvarn_mla and self._kvarn_graph:
            # FAST PATH: fused dual-source tile-decode kernel (single static launch
            # -> CUDA-graph capturable). Rotate query by H, run the kernel (int4
            # tiles + rotated fp16 pool, online softmax in rotated frame),
            # un-rotate output. See _kvarn_mla_tile_decode_kernel (cos 0.99999).
            R = self.kv_lora_rank
            ROPE = self.qk_rope_head_dim
            G = self._kvarn_group
            NB, SC, ZP, SR, RP, REC, _ = kvarn_mla_tile_layout(R, ROPE, G, self._kvarn_bits)
            Hb = self._kvarn_hadamard_act(R, q.device, q.dtype)       # [R,R] bf16
            q_lat_rot = (q[:, :, :R] @ Hb)                            # bf16 tensor-core
            q_rope = q[:, :, R:R + ROPE]
            Q = torch.cat([q_lat_rot, q_rope], dim=-1).contiguous()   # [B,H,R+ROPE] bf16
            cache_flat = kv_c_and_k_pe_cache.reshape(-1)              # uint8
            b2s = type(self)._kvarn_b2slot_t[q.device]
            O = torch.empty(B, q_num_heads, R, dtype=torch.float32, device=q.device)
            Lse = torch.empty(B, q_num_heads, dtype=torch.float32, device=q.device)
            bt = attn_metadata.decode.block_table
            seq_lens = attn_metadata.decode.seq_lens.to(torch.int32)
            BLOCK_N = int(os.environ.get("KVARN_MLA_BLOCK_N", "64"))
            warps = int(os.environ.get("KVARN_MLA_WARPS", "8"))
            stages = int(os.environ.get("KVARN_MLA_STAGES", "2"))
            # Split-K count (fixed at capture for graphs). Auto-scale with the
            # model's max context: 16 was tuned at <=8K (+3.6% short-ctx,
            # +8% @8K vs 4), but at 16K (128 blocks) low-batch decode
            # under-parallelizes -> bump to 32 (burst@16K B8: 0.82x -> 0.90x vs
            # bf16; 64 gives no further gain). Targets ~4 blocks/split at max ctx.
            # KVARN_MLA_SPLITS overrides.
            _env_nspl = os.environ.get("KVARN_MLA_SPLITS")
            if _env_nspl is not None:
                NSPL = int(_env_nspl)
            else:
                _mb = getattr(self, "_kvarn_max_blocks", None)
                if _mb is None:
                    try:
                        from vllm.config import get_current_vllm_config
                        _ml = get_current_vllm_config().model_config.max_model_len
                        _mb = (_ml + G - 1) // G
                    except Exception:
                        _mb = 999   # unknown -> assume long-ctx (more splits, safe)
                    self._kvarn_max_blocks = _mb
                NSPL = 16 if _mb <= 80 else (32 if _mb <= 256 else 64)
            if NSPL <= 1:
                _kvarn_mla_tile_decode_kernel[(B, q_num_heads)](
                    Q, cache_flat, self._kvarn_lat_pool, self._kvarn_rope_pool,
                    bt, seq_lens, b2s, O, Lse, self.scale,
                    Q.stride(0), Q.stride(1), bt.stride(0), O.stride(0), O.stride(1), Lse.stride(0),
                    self._kvarn_lat_pool.stride(0), self._kvarn_lat_pool.stride(1),
                    self._kvarn_rope_pool.stride(0), self._kvarn_rope_pool.stride(1),
                    R=R, ROPE=ROPE, G=G, NB=NB, SC=SC, ZP=ZP, SR=SR, RP=RP, REC=REC,
                    BLOCK_N=BLOCK_N, NUM_BLOCKS_LOOKUP=b2s.shape[0],
                    num_warps=warps, num_stages=stages)
            else:
                # GROUPED-HEAD split-K: all query heads share each block's dequant
                # (tensor-core dots), split-K for occupancy. stage2 = LSE-combine.
                HGROUP = 16
                n_hg = (q_num_heads + HGROUP - 1) // HGROUP
                partO = torch.empty(B, q_num_heads, NSPL, R, dtype=torch.float32, device=q.device)
                partLse = torch.empty(B, q_num_heads, NSPL, dtype=torch.float32, device=q.device)
                _kvarn_mla_grouped_stage1[(B, n_hg, NSPL)](
                    Q, cache_flat, self._kvarn_lat_pool, self._kvarn_rope_pool,
                    bt, seq_lens, b2s, partO, partLse, self.scale,
                    Q.stride(0), Q.stride(1), bt.stride(0),
                    partO.stride(0), partO.stride(1), partO.stride(2), partLse.stride(0), partLse.stride(1),
                    self._kvarn_lat_pool.stride(0), self._kvarn_lat_pool.stride(1),
                    self._kvarn_rope_pool.stride(0), self._kvarn_rope_pool.stride(1),
                    H=q_num_heads, HGROUP=HGROUP,
                    R=R, ROPE=ROPE, G=G, NB=NB, SC=SC, ZP=ZP, SR=SR, RP=RP, REC=REC,
                    BLOCK_N=min(BLOCK_N, 32),  # cap: grouped dot tiles must fit shared mem
                    NUM_SPLITS=NSPL, NUM_BLOCKS_LOOKUP=b2s.shape[0],
                    num_warps=warps, num_stages=stages)
                _kvarn_mla_splitk_stage2[(B, q_num_heads)](
                    partO, partLse, O, Lse,
                    partO.stride(0), partO.stride(1), partO.stride(2), partLse.stride(0), partLse.stride(1),
                    O.stride(0), O.stride(1), Lse.stride(0), R=R, NUM_SPLITS=NSPL)
            o = (O.to(q.dtype) @ Hb.t())                              # un-rotate (bf16 TC)
            lse = Lse.to(q.dtype)
            return o, lse

        if self._is_kvarn_mla:
            # Full-method KVarN-MLA decode (eager, correctness-first): per
            # sequence, gather rotated keys from flushed int4 tiles (dequant) +
            # staged fp16 partial blocks, dot the Hadamard-rotated query, softmax,
            # then un-rotate the output. Triton dequant->stock-attention + graphs
            # come after this validates.
            R = self.kv_lora_rank
            ROPE = self.qk_rope_head_dim
            G = self._kvarn_group
            bits = self._kvarn_bits
            NB, SC, ZP, SR, RP, REC, _ = kvarn_mla_tile_layout(R, ROPE, G, bits)
            H = self._kvarn_hadamard(R, q.device, q.dtype)
            bt = attn_metadata.decode.block_table
            seq_lens = attn_metadata.decode.seq_lens
            flat = kv_c_and_k_pe_cache.view(-1, REC)
            scale_attn = self.scale

            def unpack(block_id):
                rec = flat[block_id]
                pk = rec[:NB].view(G, R // 2)
                qd = torch.empty(G, R, dtype=torch.float32, device=q.device)
                qd[:, 0::2] = (pk & 0xF).float()
                qd[:, 1::2] = (pk >> 4).float()
                sc = rec[SC:SC + R * 2].view(torch.float16).float()
                zp = rec[ZP:ZP + R * 2].view(torch.float16).float()
                pt = rec[SR:SR + G * 2].view(torch.float16).float()
                deq_rot = (qd * sc[None, :] + zp[None, :]) * pt[:, None]  # [G,R] rot
                rope = rec[RP:RP + G * ROPE * 2].view(torch.float16).reshape(G, ROPE)
                return deq_rot, rope.float()

            for b in range(B):
                slen = int(seq_lens[b].item())
                nblk = (slen + G - 1) // G
                row = bt[b]
                Krot_parts, Krope_parts = [], []
                for j in range(nblk):
                    bid = int(row[j].item())
                    n = G if (j + 1) * G <= slen else (slen - j * G)
                    if self._kvarn_graph:
                        # graph path: in-progress block lives in the sparse pool
                        # (slot>=0); flushed blocks live int4 in the cache.
                        slot = type(self)._kvarn_b2slot_dict.get(bid)
                        st = None
                    else:
                        slot = None
                        st = self._kvarn_stage.get(bid)
                    if slot is not None:                     # pooled fp16 partial
                        # pool stores ALREADY-ROTATED latent (c@H) -> no re-rotate
                        krot = self._kvarn_lat_pool[slot][:n].float()
                        krope = self._kvarn_rope_pool[slot][:n].float()
                    elif st is not None:                     # staged fp16 partial
                        krot = (st["lat"][:n].float() @ H)   # rotate to match
                        krope = st["rope"][:n].float()
                    else:                                    # flushed int4 tile
                        d, rp = unpack(bid)
                        import os as _os
                        if _os.environ.get("KVARN_MLA_DBG") and hasattr(self, "_dbg") and bid in self._dbg:
                            dref, rref = self._dbg[bid]
                            e = (d - dref).abs().max().item()
                            if not hasattr(self, "_dbg_printed"):
                                self._dbg_printed = True
                                print(f"[DBG] flushed tile {bid}: dequant vs fp16-ref "
                                      f"max_abs={e:.4f} rel={(d-dref).norm()/dref.norm():.4f}", flush=True)
                            if _os.environ.get("KVARN_MLA_DBG") == "use_fp16":
                                d, rp = dref, rref
                        krot, krope = d[:n], rp[:n]
                    Krot_parts.append(krot)
                    Krope_parts.append(krope)
                Krot = torch.cat(Krot_parts, 0)              # [slen, R] rotated
                Krope = torch.cat(Krope_parts, 0)            # [slen, ROPE]
                import os as _o2
                if _o2.environ.get("KVARN_DBG2") and not getattr(self, "_dbg2p", 0) > 12:
                    self._dbg2p = getattr(self, "_dbg2p", 0) + 1
                    bids = [int(row[j].item()) for j in range(nblk)]
                    staged = [b2 in self._kvarn_stage for b2 in bids]
                    print(f"[DBG2] b={b} slen={slen} nblk={nblk} norm={Krot.norm():.1f} "
                          f"bids={bids} staged={staged}", flush=True)
                qb = q[b].float()                            # [Hh, R+ROPE]
                q_lat_rot = qb[:, :R] @ H
                q_rope = qb[:, R:R + ROPE]
                sc = (q_lat_rot @ Krot.t() + q_rope @ Krope.t()) * scale_attn
                p = torch.softmax(sc, dim=-1)                # [Hh, slen]
                acc_rot = p @ Krot                           # [Hh, R] = o@H
                o[b] = (acc_rot @ H.t()).to(o.dtype)
                lse[b] = torch.logsumexp(sc, dim=-1).to(lse.dtype)
            return o, lse

        # For batch invariance, use only 1 split to ensure deterministic reduction
        if envs.VLLM_BATCH_INVARIANT:
            num_kv_splits = 1
        else:
            # Minimum work per split
            # hardware dependent
            min_work_per_split = 512

            ideal_splits = max(1, attn_metadata.max_seq_len // min_work_per_split)

            # use power of 2 to avoid excessive kernel instantiations
            ideal_splits = triton.next_power_of_2(ideal_splits)

            # Calculate SM-based maximum splits with occupancy multiplier
            # 2-4x allows multiple blocks per SM for latency hiding
            # hardware dependent
            occupancy_multiplier = 2
            max_splits = self._sm_count * occupancy_multiplier
            num_kv_splits = min(ideal_splits, max_splits)

        # TODO(lucas) Allocate ahead of time
        attn_logits = torch.empty(
            (
                B,
                q_num_heads,
                num_kv_splits,
                # NOTE: the +1 stores the LogSumExp (LSE) that the stage2
                # kernel uses to merge partial attention outputs across splits.
                self.kv_lora_rank + 1,
            ),
            dtype=torch.float32,
            device=q.device,
        )

        # Add a head dim of 1
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.unsqueeze(2)
        kv_c_cache = kv_c_and_k_pe_cache[..., : self.kv_lora_rank]
        PAGE_SIZE = kv_c_and_k_pe_cache.size(1)

        # Run MQA — always pass layer scales. When KV cache is
        # BF16 the kernel's `if dtype.is_fp8()` check is a no-op.
        decode_attention_fwd(
            q,
            kv_c_and_k_pe_cache,
            kv_c_cache,
            o,
            lse,
            attn_metadata.decode.block_table,
            attn_metadata.decode.seq_lens,
            attn_logits,
            num_kv_splits,
            self.scale,
            PAGE_SIZE,
            k_scale=layer._k_scale,
            v_scale=layer._k_scale,
            is_mla=True,
        )

        return o, lse
