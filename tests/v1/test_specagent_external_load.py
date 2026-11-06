# SPDX-License-Identifier: Apache-2.0
"""An observation scratch fetch must preserve an admitted external load."""

# Standard
import pickle
import threading
from types import MethodType
from types import SimpleNamespace as NS
from unittest.mock import Mock

# Third Party
import pytest
import torch

# First Party
from lmcache.integration.vllm.specagent_external_load import SpecAgentExternalLoad
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorMetadata,
    LMCacheConnectorV1Impl,
    LoadSpec,
    RequestTracker,
)


@pytest.mark.parametrize("complete", [False, True])
def test_start_load_executes_external_restore_before_scratch(complete):
    events = []
    load = SpecAgentExternalLoad("r", list(range(100)), [1] * 7, 16, 64, 99)

    def retrieve(*args, **kwargs):
        events.append("external")
        return torch.full((100,), complete, dtype=torch.bool)

    engine = NS(
        retrieve=retrieve,
        gpu_connector=NS(
            load_stream=NS(wait_event=lambda event: events.append("scratch"))
        ),
    )
    adapter = object.__new__(LMCacheConnectorV1Impl)
    meta = LMCacheConnectorMetadata(specagent_external_loads=[load])
    adapter._parent = NS(_get_connector_metadata=lambda: meta)
    adapter._manager = NS(lmcache_engine=engine)
    adapter.kv_caches = {"layer": torch.zeros(1)}
    adapter._lmcache_chunk_size = 32
    adapter.use_layerwise = False
    adapter._specagent_prefetch_lock = threading.Lock()
    adapter._specagent_prefetch_future = None
    adapter._specagent_save_futures = []
    adapter._specagent_waited_layer_groups = set()
    ctx = NS(
        attn_metadata=object(),
        additional_kwargs={"specagent_scratch_reuse_event": object()},
    )
    if complete:
        adapter.start_load_kv(ctx)
        assert events == ["external", "scratch"]
    else:
        with pytest.raises(RuntimeError, match="incomplete"):
            adapter.start_load_kv(ctx)
        assert events == ["external"]


def make_metadata(*, resumed=False, external=True, source_end=97):
    """Exercise the real metadata builder with snapshot < accepted prefix."""
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.output import (
        CachedRequestData,
        NewRequestData,
        SchedulerOutput,
    )

    tokens = list(range(source_end))
    computed = source_end - 1  # Full hit recomputes the last prompt token.
    blocks = [7, 0, 0, 4, 6, 3, 5]
    adapter = object.__new__(LMCacheConnectorV1Impl)
    adapter.kv_role = "kv_both"
    adapter.force_skip_save = False
    adapter.config = NS(priority_limit=None, save_decode_cache=False)
    adapter.load_specs = {"r": LoadSpec(64, source_end, True)}
    adapter._block_size = 16
    adapter._lmcache_chunk_size = 32
    adapter._discard_partial_chunks = True
    adapter._requests_priority = {}
    adapter._request_trackers = {}
    adapter._unfinished_requests = {
        "r": NS(all_token_ids=tokens, num_computed_tokens=computed)
    }
    out = SchedulerOutput.make_empty()
    out.num_scheduled_tokens = {"r": 1}
    out.total_num_scheduled_tokens = 1
    out.specagent_fetch_reqs = {"r": "observation_prefill"}
    out.specagent_fetch_cpu_prefix_blocks = {"r": 4}
    out.specagent_fetch_resident_logical_indices = {"r": ([0, 3, 4, 5],)}
    out.specagent_current_block_ids = {"r": (blocks,)}
    out.specagent_external_loads = {"r": (64, computed)} if external else None
    if resumed:
        # Stale tracker block IDs must be replaced by the authoritative table.
        adapter._request_trackers["r"] = RequestTracker(
            req_id="r",
            prompt_len=len(tokens),
            token_ids=tokens[:64],
            allocated_block_ids=[1, 2, 3, 4],
        )
        out.scheduled_cached_reqs = CachedRequestData(
            req_ids=["r"],
            resumed_req_ids={"r"},
            new_token_ids=[[tokens[-1]]],
            all_token_ids={"r": tokens},
            new_block_ids=[(blocks,)],
            num_computed_tokens=[computed],
            num_output_tokens=[0],
        )
    else:
        out.scheduled_new_reqs = [
            NewRequestData(
                req_id="r",
                prompt_token_ids=tokens,
                mm_features=[],
                sampling_params=SamplingParams(max_tokens=1),
                pooling_params=None,
                block_ids=(blocks,),
                num_computed_tokens=computed,
                lora_request=None,
            )
        ]
    return adapter, out


@pytest.mark.parametrize("resumed", [False, True])
@pytest.mark.parametrize("source_end", [97, 100])
def test_metadata_preserves_original_source_despite_scratch_save_trim(
    resumed, source_end
):
    adapter, out = make_metadata(resumed=resumed, source_end=source_end)
    meta = adapter.build_connector_meta(out)
    meta = pickle.loads(pickle.dumps(meta))  # Scheduler -> worker boundary.
    assert len(meta.specagent_external_loads) == 1
    load = meta.specagent_external_loads[0]
    assert (load.start, load.end) == (64, source_end - 1)
    assert load.token_ids == list(range(source_end))
    assert load.block_ids == out.specagent_current_block_ids["r"][0]
    assert meta.requests[0].load_spec.lmcache_cached_tokens == 64
    assert len(meta.requests[0].token_ids) < source_end  # Save trims separately.
    assert out.specagent_fetch_cpu_prefix_blocks == {"r": 4}
    assert "r" not in adapter.load_specs


def test_no_admitted_external_load_does_not_create_restore():
    adapter, out = make_metadata(external=False)
    assert adapter.build_connector_meta(out).specagent_external_loads == []


@pytest.mark.parametrize(
    "invalid", [None, LoadSpec(64, 97, False), LoadSpec(0, 64, True)]
)
def test_metadata_rejects_unbacked_load_promise(invalid):
    adapter, out = make_metadata()
    adapter.load_specs = {"r": invalid}
    with pytest.raises(RuntimeError, match="lacks an admitted LoadSpec"):
        adapter.build_connector_meta(out)


@pytest.mark.parametrize("retrieved_end", [64, 96, 100])
def test_non_layerwise_retrieve_checks_exact_coverage(retrieved_end):
    load = SpecAgentExternalLoad("r", list(range(100)), [1] * 7, 16, 67, 99)
    mask = torch.ones(retrieved_end, dtype=torch.bool)
    engine = NS(retrieve=Mock(return_value=mask))
    if retrieved_end < 99:
        with pytest.raises(RuntimeError, match="incomplete"):
            load.restore(engine, [torch.zeros(1)], 32, layerwise=False)
    else:
        assert load.restore(engine, [torch.zeros(1)], 32, layerwise=False) == 32
    actual_mask = engine.retrieve.call_args.args[1]
    assert actual_mask.nonzero().flatten().tolist() == list(range(64, 100))
    slots = engine.retrieve.call_args.kwargs["slot_mapping"]
    assert (slots >= 0).nonzero().flatten().tolist() == list(range(67, 99))


def test_missing_cpu_source_fails_before_gpu_transfer():
    transfer = Mock()
    release = Mock()
    engine = NS(
        native_bulk_spec=True,
        is_healthy=lambda: True,
        plan_sparse_layerwise_chunks=lambda *args: [[(64, 96, "a"), (96, 100, "b")]],
        storage_manager=NS(
            batched_get=lambda *args, **kw: [NS(ref_count_down=release), None]
        ),
        gpu_connector=NS(specagent_restore_native_sparse=transfer),
    )
    load = SpecAgentExternalLoad("r", list(range(100)), [1] * 7, 16, 64, 99)
    with pytest.raises(RuntimeError, match="incomplete"):
        load.restore(engine, [torch.zeros(1)], 32, layerwise=True)
    transfer.assert_not_called()
    release.assert_called_once()


def test_null_external_destination_fails_before_cpu_lookup():
    load = SpecAgentExternalLoad(
        "r", list(range(100)), [1, 0, 0, 4, 0, 3, 5], 16, 64, 99
    )
    with pytest.raises(RuntimeError, match="null destination"):
        load.restore(NS(), [torch.zeros(1)], 32, layerwise=True)


@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("source_end", [97, 100])
def test_gpu_native_restore_exact_values_and_untouched_slots(layout, source_end):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemGPUConnectorV2
    from lmcache.v1.gpu_connector.specagent_native_bulk_connector import (
        SpecAgentNativeBulkGPUConnector,
    )
    from lmcache.v1.memory_management import MemoryFormat
    from lmcache.v1.cache_engine import LMCacheEngine
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.metadata import LMCacheMetadata
    from lmcache.v1.token_database import ChunkedTokenDatabase

    adapter, out = make_metadata(source_end=source_end)
    load = adapter.build_connector_meta(out).specagent_external_loads[0]
    layers, heads, dim, block = 3, 2, 128, 16
    connector = object.__new__(SpecAgentNativeBulkGPUConnector)
    connector.bulk_connector = VLLMPagedMemGPUConnectorV2(
        heads * dim,
        layers,
        device=torch.device("cuda"),
        use_gpu=False,
        layout_hints={"kv_layout": layout},
    )
    caches = []
    for _ in range(layers):
        shape = (
            (2, 8, block, heads, dim) if layout == "NHD" else (2, 8, heads, block, dim)
        )
        physical = torch.full(shape, -7, device="cuda", dtype=torch.bfloat16)
        caches.append(physical if layout == "NHD" else physical.permute(0, 1, 3, 2, 4))
    sources = [
        torch.randn((2, layers, n, heads * dim), dtype=torch.bfloat16).pin_memory()
        for n in (32, source_end - 96)
    ]
    releases = []
    owners = [
        NS(
            tensor=s,
            metadata=NS(fmt=MemoryFormat.KV_2LTD),
            ref_count_down=lambda: releases.append(True),
        )
        for s in sources
    ]

    cfg = LMCacheEngineConfig.from_legacy(
        chunk_size=32, backend="cpu", save_unfull_chunk=True
    )
    metadata = LMCacheMetadata(
        "test", 1, 1, 0, 0, torch.bfloat16, (layers, 2, 32, heads, dim)
    )
    db = ChunkedTokenDatabase(cfg, metadata)
    saved_keys = [
        key
        for start, _, key in db.process_tokens(tokens=list(range(source_end)))
        if start >= 64
    ]
    saved = dict(zip(saved_keys, owners, strict=True))
    engine = NS(
        native_bulk_spec=True,
        is_healthy=lambda: True,
        token_database=db,
        storage_manager=NS(
            batched_get=lambda keys, **kw: [saved.get(key) for key in keys]
        ),
        gpu_connector=connector,
    )
    engine.plan_sparse_layerwise_chunks = MethodType(
        LMCacheEngine.plan_sparse_layerwise_chunks, engine
    )
    assert load.restore(engine, caches, 32, layerwise=True) == source_end - 65
    assert len(releases) == 2
    for layer, cache in enumerate(caches):
        expected = torch.full((2, 128, heads, dim), -7, dtype=torch.bfloat16)
        for token in range(64, source_end - 1):
            chunk = 0 if token < 96 else 1
            offset = token - (64 if chunk == 0 else 96)
            dest = load.block_ids[token // block] * block + token % block
            expected[:, dest] = sources[chunk][:, layer, offset].reshape(2, heads, dim)
        torch.testing.assert_close(
            cache.cpu().reshape_as(expected), expected, rtol=0, atol=0
        )
