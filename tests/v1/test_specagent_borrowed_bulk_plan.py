# SPDX-License-Identifier: Apache-2.0
"""CPU chunk leases survive cancellation, overlap and deferred GPU completion."""

# Standard
from concurrent.futures import Future
from types import SimpleNamespace

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1 import cache_engine, specagent_bulk_view
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)


class Allocator:
    def __init__(self):
        self.freed = []

    def free(self, obj, allocator_type=None):
        self.freed.append(obj)
        obj.invalidate()


def setup_engine():
    allocator = Allocator()
    keys = [CacheEngineKey("test", 1, 0, i, torch.bfloat16) for i in range(2)]
    objects = {}
    for key in keys:
        tensor = (
            torch.arange(2 * 4 * 17 * 8, dtype=torch.float32)
            .reshape(2, 4, 17, 8)
            .to(torch.bfloat16)
        )
        objects[key] = TensorMemoryObj(
            tensor.view(torch.uint8).flatten(),
            MemoryObjMetadata(
                tensor.shape,
                tensor.dtype,
                tensor.data_ptr(),
                tensor.numel() * 2,
                1,
                fmt=MemoryFormat.KV_2LTD,
            ),
            allocator,
        )

    def get(requested):
        rows = [objects.get(k) for k in requested]
        for row in rows:
            if row is not None:
                row.ref_count_up()
        return rows

    engine = LMCacheEngine.__new__(LMCacheEngine)
    engine.native_bulk_spec = True
    engine.num_layers = 4
    engine.retrieve_locations = None
    engine.storage_manager = SimpleNamespace(
        batched_contains=lambda requested, _: (
            len(requested),
            {"LocalCPUBackend": requested},
        ),
        storage_backends={"LocalCPUBackend": SimpleNamespace(batched_get_blocking=get)},
    )
    plans = [[(0, 17, keys[0])], [(17, 34, keys[1])]]
    return engine, allocator, objects, plans


def test_unclaimed_segments_release_only_their_chunk_owners():
    engine, allocator, objects, plans = setup_engine()
    batch = engine.prepare_sparse_layerwise_plans(plans)
    first = batch.claim(0)
    assert all(o.get_ref_count() == 2 for o in objects.values())
    for obj in objects.values():
        obj.ref_count_down()  # evict cache entries while the read owns them
    batch.release_unclaimed()
    assert len(allocator.freed) == 1
    assert first.source_owners[0].is_valid()
    for i, layer in enumerate(first.memory_objs_by_layer):
        assert torch.equal(
            layer[0].tensor, first.source_owners[0].tensor[:, i].permute(1, 0, 2)
        )
    with pytest.raises(RuntimeError, match="already claimed"):
        batch.claim(0)
    first.release_source_owners()
    first.release_source_owners()
    assert len(allocator.freed) == 2


def test_descriptor_reuse_does_not_acquire_extra_refs_and_tracks_replacement():
    engine, allocator, objects, plans = setup_engine()
    obj = next(iter(objects.values()))
    first = specagent_bulk_view.borrow_bulk_layer_views([obj], 4)
    second = specagent_bulk_view.borrow_bulk_layer_views([obj], 4)
    assert first[0][0] is second[0][0]
    assert obj.get_ref_count() == 1
    obj.raw_data = torch.zeros_like(obj.raw_data)
    third = specagent_bulk_view.borrow_bulk_layer_views([obj], 4)
    assert third[0][0] is not first[0][0]
    assert torch.count_nonzero(third[0][0].tensor) == 0
    for owner in objects.values():
        owner.ref_count_down()
    assert len(allocator.freed) == 2


def test_missing_owner_rolls_back_acquired_chunk_refs():
    engine, allocator, objects, plans = setup_engine()
    key = list(objects)[1]
    missing = objects.pop(key)
    with pytest.raises(RuntimeError, match="evicted"):
        engine.prepare_sparse_layerwise_plans(plans)
    assert next(iter(objects.values())).get_ref_count() == 1
    assert missing.get_ref_count() == 1
    missing.ref_count_down()
    next(iter(objects.values())).ref_count_down()
    assert len(allocator.freed) == 2


@pytest.mark.parametrize("deferred", [False, True])
def test_retrieve_transfers_chunk_lease_or_uses_independent_fallback(
    monkeypatch, deferred
):
    engine, allocator, objects, plans = setup_engine()
    received = []

    def consume(starts, ends, **kwargs):
        for _ in range(4):
            rows = yield
            received.extend(rows)
            assert all(o.is_valid() for o in objects.values())
        yield
        yield

    monkeypatch.setattr(cache_engine, "assert_layerwise_gpu_connector", lambda _: None)
    engine.gpu_connector = SimpleNamespace(batched_to_gpu=consume)
    engine.is_healthy = lambda: True
    engine.stats_monitor = SimpleNamespace(
        on_retrieve_request=lambda _: 1, on_retrieve_finished=lambda *args: None
    )
    engine.save_only_first_rank = True
    engine.metadata = SimpleNamespace(is_first_rank=lambda: False)
    batch = engine.prepare_sparse_layerwise_plans([plans[0]])
    future = Future()
    future.set_result(batch)
    gpu_owner = {} if deferred else None
    list(
        engine.retrieve_layer(
            list(range(17)),
            specagent_expected_tokens=17,
            specagent_prefix_tokens=0,
            specagent_chunk_plan=plans[0],
            specagent_prepared_plan_future=future,
            specagent_deferred_source_owner=gpu_owner,
        )
    )
    first = next(iter(objects.values()))
    assert len(received) == 4
    if deferred:
        assert gpu_owner["memory_objs"] == [first]
        first.ref_count_down()  # eviction, still protected by deferred owner
        assert first.is_valid()
        for owner in gpu_owner.pop("memory_objs"):
            owner.ref_count_down()  # GPU completion
        assert not first.is_valid()
    else:
        assert all(isinstance(r, specagent_bulk_view.BulkLayerView) for r in received)
        assert all(not r.is_valid() for r in received)
        assert first.get_ref_count() == 1
        first.ref_count_down()
    list(objects.values())[1].ref_count_down()
    assert len(allocator.freed) == 2


def test_claimed_segment_abandoned_before_consumer_releases_owners():
    engine, allocator, objects, plans = setup_engine()
    batch = engine.prepare_sparse_layerwise_plans(plans)
    segment = batch.claim(0)
    for owner in objects.values():
        owner.ref_count_down()
    batch.release_unclaimed()
    assert len(allocator.freed) == 1
    del segment
    assert len(allocator.freed) == 2
