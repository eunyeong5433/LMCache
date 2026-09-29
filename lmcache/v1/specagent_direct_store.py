# SPDX-License-Identifier: Apache-2.0
"""Synchronous paged D2H into existing layer-keyed CPU cache objects."""
# Standard
from typing import Any

# Third Party
import torch
import triton
import triton.language as tl


@triton.jit
def _store_paged_layers_kernel(desc, slots, H: tl.constexpr, HEAD: tl.constexpr,
           BLOCK: tl.constexpr, BF16: tl.constexpr, TILE: tl.constexpr):
    plan = tl.program_id(1)
    row = desc + plan * 10
    src = tl.load(row + 0).to(tl.pointer_type(tl.bfloat16 if BF16 else tl.float16))
    dst = tl.load(row + 1).to(tl.pointer_type(tl.bfloat16 if BF16 else tl.float16))
    start = tl.load(row + 2)
    count = tl.load(row + 3)
    s0 = tl.load(row+5)
    s1 = tl.load(row+6)
    s2 = tl.load(row+7)
    s3 = tl.load(row+8)
    s4 = tl.load(row+9)
    idx = tl.program_id(0) * TILE + tl.arange(0, TILE)
    token = idx // (2 * H)
    kv = (idx // H) % 2
    feature = idx % H
    slot = tl.load(slots + start + token, token < count, other=-1)
    valid = (token < count) & (slot >= 0)
    src_idx = kv * s0 + (slot // BLOCK) * s1 + (slot % BLOCK) * s2 + (feature // HEAD) * s3 + (feature % HEAD) * s4
    value = tl.load(src + src_idx, valid, other=0)
    # Missing slots are deliberately untouched, matching sparse tail-merge use.
    tl.store(dst + idx, value, valid)


def store_paged_layers(
    layers: list[list[Any]], starts: list[int], ends: list[int],
    caches: list[torch.Tensor], slots: torch.Tensor,
    stream: torch.cuda.Stream, offset: int = 0,
) -> None:
    """Copy paged KV into existing pinned, token-major CPU layer objects.

    Args:
        layers: CPU MemoryObj destinations in layer/chunk order.
        starts: Absolute logical start of each saved chunk.
        ends: Exclusive logical chunk ends, including an exact partial tail.
        caches: GPU views shaped [2, blocks, block_size, heads, head_size].
        slots: Contiguous int64 CUDA mapping; negative slots are left untouched.
        stream: Store stream; completion is awaited even after submission errors.
        offset: Logical position represented by slots[0].

    Caller retains every destination until this function returns. CPU cache
    keys and proof publication remain the engine's responsibility.
    """
    if not starts:
        return
    dtype = caches[0].dtype
    assert dtype in (torch.bfloat16, torch.float16)
    assert slots.is_cuda and slots.dtype == torch.int64 and slots.is_contiguous()
    rows = []
    for objects, cache in zip(layers, caches, strict=True):
        assert cache.ndim == 5 and cache.shape[0] == 2 and cache.dtype == dtype
        hidden = cache.shape[-2] * cache.shape[-1]
        for obj, start, end in zip(objects, starts, ends, strict=True):
            dst = obj.tensor
            assert dst.is_pinned() and dst.is_contiguous() and dst.dtype == dtype
            assert tuple(dst.shape) == (end-start, 2, hidden)
            assert 0 <= start-offset < end-offset <= slots.numel()
            rows.append([cache.data_ptr(), dst.data_ptr(), start-offset, end-start, 0, *cache.stride()])
    descriptors = None
    current = torch.cuda.current_stream(caches[0].device)
    try:
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            descriptors = torch.tensor(rows, dtype=torch.int64, device=slots.device)
            _store_paged_layers_kernel[(triton.cdiv(max(e-s for s,e in zip(starts,ends))*2*hidden,1024),len(rows))](
                descriptors, slots, hidden, caches[0].shape[-1], caches[0].shape[2],
                dtype == torch.bfloat16,1024)
    finally:
        stream.synchronize()
