# SPDX-License-Identifier: Apache-2.0
"""Native bulk storage must publish proof only after completed CPU storage."""
import builtins
from contextlib import nullcontext
from types import SimpleNamespace as NS
import pytest
import torch
from lmcache.utils import CacheEngineKey
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.memory_management import MemoryFormat


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("failure", [None, "copy", "publish"])
def test_bulk_proof_follows_copy_and_publication(monkeypatch, failure, native):
    monkeypatch.setenv("SPECAGENT_BULK_CPU_PROOF", "1")
    monkeypatch.delenv("SPECAGENT_DISABLE_CPU_STORE_PROOF", raising=False)
    ledger = set()
    monkeypatch.setattr(builtins, "_specagent_cpu_saved_chunks", ledger, raising=False)
    monkeypatch.setattr(builtins, "_specagent_cpu_store_hash_snapshots", [], raising=False)
    stages = []
    from lmcache.v1 import cache_engine
    prepare = cache_engine._specagent_prepare_cpu_store_proofs
    def observe_prepare(*args, **kwargs):
        assert not ledger
        stages.append("prepare")
        return prepare(*args, **kwargs)
    monkeypatch.setattr(cache_engine, "_specagent_prepare_cpu_store_proofs", observe_prepare)
    before_copy = ["prepare"] if native else []
    def copy(*args, **kwargs):
        assert not ledger and stages == before_copy
        stages.append("copy")
        if failure == "copy":
            raise RuntimeError("copy failed")
    def put(*args, **kwargs):
        assert not ledger and stages == before_copy + ["copy"]
        stages.append("publish")
        if failure == "publish":
            raise RuntimeError("publish failed")
    e = object.__new__(LMCacheEngine)
    e.native_bulk_spec = native
    e.is_healthy = lambda: True
    e._is_passive = lambda: False
    e.is_frozen = lambda: False
    e._get_req_id = lambda kwargs: "test"
    e._log_kvcache_for_check = lambda **kwargs: None
    e.gpu_connector = NS(batched_from_gpu=copy)
    e.storage_manager = NS(allocate=lambda *a, **kw: NS(get_size=lambda: 100), batched_put=put)
    e.metadata = NS(get_shapes=lambda n: torch.Size([2, 3, n, 8]), get_dtypes=lambda: torch.bfloat16)
    e.config = NS(get_extra_config_value=lambda *args: False)
    e.fmt = MemoryFormat.KV_2LTD
    e.store_location = "LocalCPUBackend"
    e.kv_events_enabled = False
    def chunks(tokens, *args, **kwargs):
        for start in range(0, len(tokens), 16):
            end = min(start + 16, len(tokens))
            yield start, end, CacheEngineKey("m", 1, 0, end, torch.bfloat16)
    e.token_database = NS(process_tokens=chunks)
    stats = NS(profile_process_tokens=nullcontext, profile_from_gpu=nullcontext,
               profile_put=nullcontext, time_to_store=lambda: 1.,
               process_tokens_time=0., from_gpu_time=0., put_time=0.)
    e.stats_monitor = NS(on_store_request=lambda n: stats, on_store_finished=lambda *a: None)
    tokens = list(range(35))
    if failure:
        with pytest.raises(RuntimeError):
            e.store(tokens, req_id="test")
        assert not ledger
    else:
        e.store(tokens, req_id="test")
        assert stages == (
            ["prepare", "copy", "publish"] if native
            else ["copy", "publish", "prepare"]
        )
        assert ledger == {(n, hash(tuple(tokens[:n]))) for n in (16, 32, 35)}


def test_bulk_live_coverage_uses_chunk_owner_and_stops_at_hole():
    e = object.__new__(LMCacheEngine)
    e.use_layerwise = False
    e.grouped_layers = False
    e.num_layers = 48
    e.retrieve_locations = ["LocalCPUBackend"]
    calls = []
    def contains(key, locations):
        calls.append(key)
        return "LocalCPUBackend" if key.chunk_hash == 16 else None
    e.storage_manager = NS(contains=contains)
    def chunks(tokens, **kwargs):
        for start in range(0, len(tokens), 16):
            end = min(start + 16, len(tokens))
            yield start, end, CacheEngineKey("m", 1, 0, end, torch.bfloat16)
    e.token_database = NS(chunk_size=16, process_tokens=chunks)
    assert e.specagent_live_saved_tokens(list(range(48))) == 16
    assert len(calls) == 2
