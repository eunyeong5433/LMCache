# SPDX-License-Identifier: Apache-2.0
"""Disjoint native chunks share one retriever without copying their holes."""

# Standard
from concurrent.futures import Future
from types import SimpleNamespace

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemLayerwiseGPUConnector
from lmcache.v1.memory_management import MemoryFormat, MemoryObjMetadata, TensorMemoryObj


@pytest.mark.no_shared_allocator
@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_disjoint_native_chunks_one_retriever_exact_and_leased(monkeypatch, layout):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    monkeypatch.setenv("SPECAGENT_DIRECT_HOST_SCATTER", "1")
    monkeypatch.setenv("SPECAGENT_NATIVE_LAYER_PRODUCER", "1")
    monkeypatch.delenv("SPECAGENT_BW_PROFILE", raising=False)
    layers, heads, dim = 3, 4, 128
    freed = []

    def free(obj, allocator_type=None):
        freed.append(obj)
        obj.invalidate()

    owners = []
    for n in (256, 17):
        tensor = torch.randn((2, layers, n, heads * dim), dtype=torch.bfloat16).pin_memory()
        owners.append(TensorMemoryObj(
            tensor.view(torch.uint8).flatten(),
            MemoryObjMetadata(tensor.shape, tensor.dtype, tensor.data_ptr(),
                              tensor.numel() * 2, 1, fmt=MemoryFormat.KV_2LTD),
            SimpleNamespace(free=free),
        ))
    keys = [CacheEngineKey("test", 1, 0, i, torch.bfloat16) for i in range(2)]
    objects = dict(zip(keys, owners, strict=True))

    def get(requested):
        rows = [objects[key] for key in requested]
        for row in rows:
            row.ref_count_up()
        return rows

    connector = VLLMPagedMemLayerwiseGPUConnector(
        heads * dim, layers, use_gpu=True, chunk_size=256, dtype=torch.bfloat16,
        device=torch.device("cuda"), layout_hints={"kv_layout": layout},
    )
    connector.specagent_prepare_native_layers(0, layers - 1)
    engine = LMCacheEngine.__new__(LMCacheEngine)
    engine.native_bulk_spec = True
    engine.num_layers = layers
    engine.retrieve_locations = None
    engine.storage_manager = SimpleNamespace(
        batched_contains=lambda requested, _: (len(requested), {"LocalCPUBackend": requested}),
        storage_backends={"LocalCPUBackend": SimpleNamespace(batched_get_blocking=get)},
    )
    engine.gpu_connector = connector
    engine.is_healthy = lambda: True
    engine.stats_monitor = SimpleNamespace(
        on_retrieve_request=lambda _: 1, on_retrieve_finished=lambda *args: None,
    )
    engine.save_only_first_rank = True
    engine.metadata = SimpleNamespace(is_first_rank=lambda: False)
    # A large gap must not become part of the fetch or the returned token count.
    starts, ends = [256, 65536], [512, 65553]
    plan = list(zip(starts, ends, keys, strict=True))
    batch = engine.prepare_sparse_layerwise_plans([plan])
    future = Future()
    future.set_result(batch)
    shape = ((2, 32, 16, heads, dim) if layout == "NHD"
             else (2, 32, heads, 16, dim))
    targets = [torch.full(shape, -7, dtype=torch.bfloat16, device="cuda")
               for _ in range(layers)]
    slot_cpu = torch.zeros(ends[-1], dtype=torch.int64)
    destinations = torch.arange(32, 305, dtype=torch.int64).flip(0)
    slot_cpu[256:512] = destinations[:256]
    slot_cpu[65536:65553] = destinations[256:]
    slots = slot_cpu.cuda()
    ready = [None] * layers
    expected = []
    for layer in range(layers):
        e = torch.full((2, 512, heads * dim), -7, dtype=torch.bfloat16)
        for i, owner in enumerate(owners):
            ids = destinations[:256] if i == 0 else destinations[256:]
            e[:, ids] = owner.tensor[:, layer]
        expected.append(e)
    results = list(engine.retrieve_layer(
        list(range(ends[-1])), kvcaches=targets, slot_mapping=slots,
        specagent_expected_tokens=273, specagent_prefix_tokens=256,
        specagent_chunk_plan=plan, specagent_prepared_plan_future=future,
        specagent_layer_events=ready, specagent_defer_layer_submit=True,
        specagent_deferred_source_owner={}, sync=True,
    ))
    assert [r for r in results if r is not None] == [273, 273]
    assert all(len(plans) == 1 for plans in connector.specagent_peek_graph_plans())
    assert ready == [None] * layers
    for owner in owners:
        assert owner.get_ref_count() == 2
        owner.ref_count_down()  # Cache eviction cannot invalidate an in-flight source.
    assert not freed
    for layer in range(layers):
        assert connector.specagent_flush_deferred_layer(layer)
        ready[layer].synchronize()
        physical = targets[layer]
        logical = physical if layout == "NHD" else physical.permute(0, 1, 3, 2, 4)
        torch.testing.assert_close(logical.cpu().reshape_as(expected[layer]),
                                   expected[layer], rtol=0, atol=0)
    connector.specagent_reap_completed_plans()
    assert len(freed) == 2
    connector.specagent_reap_completed_plans()
    assert len(freed) == 2
