# SPDX-License-Identifier: Apache-2.0

import threading

from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl
from lmcache.v1.cache_engine import (
    LMCacheEngine,
    SpecAgentProgressiveLayerwiseBatch,
    SpecAgentProgressiveLayerwiseSegment,
)


def test_grouped_prefetch_only_advances_at_group_boundary() -> None:
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector.num_layers = 64
    connector.current_layer = 0
    connector.layerwise_retrievers = [object()]
    connector._specagent_layerwise_meta = [
        {"is_specagent": True, "is_paged_restore": False}
    ]
    connector._specagent_layer_group_size = 4
    calls: list[tuple[int, object | None]] = []
    connector._specagent_request_async_prefetch = (
        lambda target, stream_wait_event=None: calls.append(
            (target, stream_wait_event)
        )
    )

    group0_done = object()
    for layer in range(4):
        connector.prefetch_next_layer_load(
            f"model.layers.{layer}.self_attn.attn", group0_done
        )

    assert calls == [(11, group0_done)]

    group1_done = object()
    for layer in range(4, 8):
        connector.prefetch_next_layer_load(
            f"model.layers.{layer}.self_attn.attn", group1_done
        )

    assert calls == [(11, group0_done), (15, group1_done)]


def test_progressive_plan_publishes_and_releases_each_lease_once() -> None:
    class MemoryObj:
        def __init__(self) -> None:
            self.releases = 0

        def ref_count_down(self) -> None:
            self.releases += 1

    batch = SpecAgentProgressiveLayerwiseBatch(1)
    segment = SpecAgentProgressiveLayerwiseSegment(
        starts=[0],
        ends=[256],
        keys_by_chunk=[[]],
        location="cpu",
        num_layers=2,
    )
    first = MemoryObj()
    second = MemoryObj()

    def publish() -> None:
        batch.publish_segments([segment])
        segment.publish(0, [first])
        segment.publish(1, [second])

    worker = threading.Thread(target=publish)
    worker.start()
    claimed = batch.claim(0)
    assert claimed is segment
    assert claimed.take_layer(0) == [first]
    claimed.release_pending()
    worker.join()

    # The consumer owns a taken lease; release it explicitly.  The segment
    # owns and releases the never-taken layer exactly once.
    first.ref_count_down()
    assert first.releases == 1
    assert second.releases == 1


def test_progressive_engine_resolves_layer_major_and_segments() -> None:
    class MemoryObj:
        def __init__(self, name: str) -> None:
            self.name = name
            self.releases = 0

        def ref_count_down(self) -> None:
            self.releases += 1

    keys_by_chunk = [["a0", "a1"], ["b0", "b1"], ["c0", "c1"]]
    objects = {key: MemoryObj(key) for row in keys_by_chunk for key in row}

    class Backend:
        def batched_get_blocking(self, keys):
            return [objects[key] for key in keys]

    class Storage:
        storage_backends = {"cpu": Backend()}

    engine = LMCacheEngine.__new__(LMCacheEngine)
    engine.num_layers = 2
    engine.storage_manager = Storage()
    engine.bulk_validate_sparse_layerwise_plan = lambda _plans: (
        [0, 256, 512],
        [256, 512, 768],
        keys_by_chunk,
        "cpu",
    )
    plans = [[object(), object()], [object()]]
    batch = SpecAgentProgressiveLayerwiseBatch(2)
    engine.prepare_sparse_layerwise_plans_progressive(plans, batch)

    first = batch.claim(0)
    second = batch.claim(1)
    assert first is not None and second is not None
    assert [obj.name for obj in first.take_layer(0)] == ["a0", "b0"]
    assert [obj.name for obj in first.take_layer(1)] == ["a1", "b1"]
    assert [obj.name for obj in second.take_layer(0)] == ["c0"]
    assert [obj.name for obj in second.take_layer(1)] == ["c1"]


def test_progressive_claimed_segment_observes_worker_failure() -> None:
    batch = SpecAgentProgressiveLayerwiseBatch(1)
    segment = SpecAgentProgressiveLayerwiseSegment(
        starts=[0],
        ends=[256],
        keys_by_chunk=[[]],
        location="cpu",
        num_layers=1,
    )
    batch.publish_segments([segment])
    claimed = batch.claim(0)
    assert claimed is segment
    error = RuntimeError("prepare failed")
    segment.fail(error)
    batch.fail(error)
    try:
        claimed.take_layer(0)
    except RuntimeError as caught:
        assert caught is error
    else:
        raise AssertionError("claimed segment blocked past worker failure")
