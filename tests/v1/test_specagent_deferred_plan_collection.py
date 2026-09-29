# SPDX-License-Identifier: Apache-2.0
"""Deferred source collection must not select a CUDA stream per layer."""

# Standard
from types import SimpleNamespace
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemLayerwiseGPUConnector
from lmcache.v1.memory_management import MemoryFormat
from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl


@pytest.mark.no_shared_allocator
@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("graph_collection", [False, True])
def test_collect_without_stream_then_flush_exact(
    monkeypatch: pytest.MonkeyPatch, layout: str, graph_collection: bool
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    monkeypatch.setenv("SPECAGENT_DIRECT_HOST_SCATTER", "1")
    monkeypatch.delenv("SPECAGENT_BW_PROFILE", raising=False)
    monkeypatch.setenv("SPECAGENT_NATIVE_LAYER_PRODUCER", "1")
    layers, heads, dim = 2, 2, 128
    connector = VLLMPagedMemLayerwiseGPUConnector(
        heads * dim, layers, use_gpu=True, chunk_size=256, dtype=torch.bfloat16,
        device=torch.device("cuda"), layout_hints={"kv_layout": layout},
    )
    shape = ((2, 2, 16, heads, dim) if layout == "NHD"
             else (2, 2, heads, 16, dim))
    targets = [torch.full(shape, -7, dtype=torch.bfloat16, device="cuda")
               for _ in range(layers)]
    slots = torch.arange(8, dtype=torch.int64, device="cuda")
    sources = [[torch.randn((2, 2, heads * dim), dtype=torch.bfloat16).pin_memory()
                for _ in range(2)] for _ in range(layers)]
    connector.specagent_prepare_native_layers(0, layers - 1)
    ready = [None] * layers
    iterator = connector.batched_to_gpu(
        [0, 6], [2, 8], kvcaches=targets, slot_mapping=slots, sync=True,
        specagent_layer_events=ready, specagent_defer_layer_submit=True,
        specagent_deferred_source_owner={"memory_objs": []},
    )
    next(iterator)  # Slot-map production and its stream dependency are real work.

    def forbidden_stream(*args, **kwargs):
        raise AssertionError("descriptor collection selected a CUDA stream")

    with monkeypatch.context() as guard:
        guard.setattr(torch.cuda, "stream", forbidden_stream)
        def retriever():
            yield 4
            for layer in range(layers):
                iterator.send([SimpleNamespace(tensor=t, metadata=SimpleNamespace(
                    fmt=MemoryFormat.KV_T2D)) for t in sources[layer]])
                yield None
            list(iterator)

        if graph_collection:
            # Exercise the adapter's internal producer boundary: completed
            # collection must still permit the ordinary flush fallback.
            adapter = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
            adapter._manager = SimpleNamespace(lmcache_engine=SimpleNamespace(
                gpu_connector=connector))
            adapter.num_layers = layers
            adapter.layerwise_retrievers = [retriever()]
            adapter._specagent_layerwise_meta = [dict(
                is_specagent=True, expected_tokens=4, layer_events=ready,
                defer_layer_submit=True, req_id="collection-test")]
            adapter._specagent_prefetch_scan_layer = 0
            adapter._specagent_prefetch_cond = threading.Condition()
            adapter._specagent_advance_retrievers_until(layers - 1, collect_only=True)
        else:
            list(retriever())
        for layer in range(layers):
            assert connector.specagent_has_deferred_layer(layer, ready)
            assert ready[layer] is None  # Collection cannot publish a copy event.
    # The actual copy still uses the load stream and publishes completion.
    for layer in range(layers):
        assert connector.specagent_flush_deferred_layer(layer)
        assert ready[layer] is not None
        ready[layer].synchronize()
        target = targets[layer]
        logical = target if layout == "NHD" else target.permute(0, 1, 3, 2, 4)
        actual = logical.cpu().reshape(2, 32, heads * dim)
        expected = torch.full_like(actual, -7)
        expected[:, :2] = sources[layer][0].permute(1, 0, 2)
        expected[:, 6:8] = sources[layer][1].permute(1, 0, 2)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    list(iterator)
