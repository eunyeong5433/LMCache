# SPDX-License-Identifier: Apache-2.0
"""Blocking paged restore using the existing layer-keyed CPU backup."""

# Standard
from typing import Any

# Third Party
import torch


def restore_paged_bulk(
    engine: Any,
    tokens: list[int],
    segments: list[tuple[int, int]],
    write_ranges: list[tuple[int, int]],
    kvcaches: list[torch.Tensor],
    slot_mapping: torch.Tensor,
    request_configs: dict | None = None,
) -> int:
    """Restore selected paged slots and release CPU leases after completion.

    ``segments`` covers CPU chunks; ``write_ranges`` limits actual writes so
    chunk alignment never overwrites resident KV. Missing CPU data is an
    error, not permission to continue with incomplete GPU KV.
    """
    if not segments:
        if write_ranges:
            raise ValueError("Bulk restore writes require source segments")
        return 0
    previous_end = 0
    for start, end in segments:
        if start < previous_end or start >= end or end > len(tokens):
            raise ValueError("Invalid bulk restore source ranges")
        previous_end = end
    for start, end in write_ranges:
        if not any(left <= start < end <= right for left, right in segments):
            raise ValueError("Bulk restore writes are not covered by CPU sources")
    if not engine.is_healthy():
        raise RuntimeError("Cannot restore paged KV from unhealthy LMCache")
    plans = engine.plan_sparse_layerwise_chunks(tokens, segments, request_configs)
    for (start, end), plan in zip(segments, plans, strict=True):
        cursor = start
        for chunk_start, chunk_end, _ in plan:
            if chunk_start != cursor or chunk_end > end:
                raise RuntimeError("Bulk restore CPU plan has a coverage gap")
            cursor = chunk_end
        if cursor != end:
            raise RuntimeError("Bulk restore CPU plan does not cover requested range")
    if getattr(engine, "native_bulk_spec", False):
        flat_plan = [chunk for plan in plans for chunk in plan]
        owners = engine.storage_manager.batched_get(
            [key for _, _, key in flat_plan], location="LocalCPUBackend",
        )
        acquired = [owner for owner in owners if owner is not None]
        try:
            if len(owners) != len(flat_plan) or len(acquired) != len(flat_plan):
                raise RuntimeError("Native sparse restore CPU backup is incomplete")
            engine.gpu_connector.specagent_restore_native_sparse(
                acquired, [start for start, _, _ in flat_plan],
                [end for _, end, _ in flat_plan], write_ranges, kvcaches, slot_mapping,
            )
            return sum(end - start for start, end in write_ranges)
        finally:
            for owner in acquired:
                owner.ref_count_down()
    prepared = engine.prepare_sparse_layerwise_plans(plans)
    if prepared is None:
        raise RuntimeError("Bulk restore CPU backup is incomplete or unsupported")
    claimed = []
    try:
        for index in range(len(plans)):
            claimed.append(prepared.claim(index))
        starts = [start for segment in claimed for start in segment.starts]
        ends = [end for segment in claimed for end in segment.ends]
        layers = [
            [obj for segment in claimed for obj in segment.memory_objs_by_layer[layer]]
            for layer in range(engine.num_layers)
        ]
        engine.gpu_connector.specagent_restore_paged_bulk(
            layers, starts, ends, write_ranges, kvcaches, slot_mapping
        )
        return sum(end - start for start, end in write_ranges)
    finally:
        # The connector synchronizes on success AND on partial submission
        # failures before any CPU source can be recycled by the allocator.
        for segment in claimed:
            for layer in segment.memory_objs_by_layer:
                for obj in layer:
                    obj.ref_count_down()
        prepared.release_unclaimed()
