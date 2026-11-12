# SPDX-License-Identifier: Apache-2.0
"""Finished-task CPU KV drop (lmcache.v1.specagent_job_free), CPU only."""

import builtins
from types import SimpleNamespace

import pytest

from lmcache.v1 import specagent_job_free as jf


class FakeStorage:
    def __init__(self, keys):
        self.keys = set(keys)
        self.calls = []

    def batched_remove(self, keys, locations=None):
        self.calls.append((list(keys), locations))
        n = 0
        for k in keys:
            if k in self.keys:
                self.keys.discard(k)
                n += 1
        return n


@pytest.fixture(autouse=True)
def fresh_stats():
    if hasattr(builtins, "_specagent_cpu_kv"):
        del builtins._specagent_cpu_kv
    yield
    if hasattr(builtins, "_specagent_cpu_kv"):
        del builtins._specagent_cpu_kv


def _two_jobs(common_len):
    reg = jf.JobKeyRegistry()
    a = list(range(common_len)) + [1000 + i for i in range(600)]
    b = list(range(common_len)) + [2000 + i for i in range(600)]
    reg.note_requests({"ra": "A", "rb": "B"}, {"ra": a, "rb": b})
    return reg


def test_common_prefix_is_the_lcp_of_first_requests():
    reg = _two_jobs(300)
    assert reg.common_prefix == 300
    assert jf.stats()["common_prefix_tokens"] == 300
    # A later job with a shorter shared part lowers the bound.
    reg.note_requests({"rc": "C"}, {"rc": list(range(100)) + [7] * 50})
    assert reg.common_prefix == 100
    # A job seen again does not change it.
    reg.note_requests({"rc2": "C"}, {"rc2": [9] * 400})
    assert reg.common_prefix == 100


def test_free_removes_own_chunks_but_keeps_common_prefix_chunks():
    reg = _two_jobs(300)
    # chunk 0 [0,256) lies inside the common prefix; chunks 1,2 do not.
    reg.register("ra", ["a0", "a1", "a2"], [256, 512, 700], [10, 10, 5])
    st = FakeStorage({"a0", "a1", "a2", "b1"})
    assert jf.stats()["live_bytes"] == 25
    n = reg.free("A", st)
    assert n == 2
    assert st.keys == {"a0", "b1"}
    assert st.calls == [(["a1", "a2"], ["LocalCPUBackend"])]
    s = jf.stats()
    assert s["freed_jobs"] == 1 and s["freed_keys"] == 2
    assert s["freed_bytes"] == 15 and s["kept_common_keys"] == 1
    assert s["live_bytes"] == 0 and s["peak_live_bytes"] == 25
    # Idempotent.
    assert reg.free("A", st) == 0
    assert jf.stats()["freed_jobs"] == 1


def test_key_stored_by_two_jobs_is_never_removed():
    reg = _two_jobs(10)
    reg.register("ra", ["k", "a1"], [600, 900], [4, 4])
    reg.register("rb", ["k", "b1"], [600, 900], [4, 4])
    assert jf.stats()["kept_shared_keys"] == 1
    st = FakeStorage({"k", "a1", "b1"})
    reg.free("A", st)
    reg.free("B", st)
    assert st.keys == {"k"}
    assert jf.stats()["live_bytes"] == 0


def test_unknown_request_and_late_keys():
    reg = _two_jobs(10)
    reg.register("nojob", ["x"], [600], [1])  # warmup etc.: not tracked
    assert reg.owner == {}
    st = FakeStorage(set())
    reg.free("A", st)
    reg.note_requests({"ra2": "A"}, {})
    reg.register("ra2", ["late"], [900], [1])
    assert jf.stats()["late_keys"] == 1
    assert "late" not in reg.owner


def test_missing_key_at_free_is_counted():
    reg = _two_jobs(10)
    reg.register("ra", ["a1", "a2"], [600, 900], [1, 1])
    st = FakeStorage({"a1"})  # a2 was LRU-evicted earlier
    assert reg.free("A", st) == 1
    assert jf.stats()["remove_missing"] == 1


def test_note_cpu_alloc_counts_only_pressure():
    jf.note_cpu_alloc(0, False)
    assert jf.stats()["evicted"] == 0
    jf.note_cpu_alloc(3, False)
    jf.note_cpu_alloc(0, True)
    s = jf.stats()
    assert s["evicted"] == 3 and s["alloc_failed"] == 1


def test_adapter_wiring_scheduler_to_worker():
    from lmcache.integration.vllm import vllm_v1_adapter as ad

    # Scheduler-side instance without __init__ (only the wiring is tested).
    sched = object.__new__(ad.LMCacheConnectorV1Impl)
    sched._specagent_finished_save_metas = {}
    req_a = SimpleNamespace(request_id="ra", job_id="A")
    req_b = SimpleNamespace(request_id="rb", job_id="B")
    sched.specagent_note_request_job(req_a)
    sched.specagent_note_request_job(req_b)
    sched.specagent_note_request_job(SimpleNamespace(request_id="x", job_id=None))

    def inner(_out, reqs):
        m = ad.LMCacheConnectorMetadata()
        for rid, toks in reqs:
            m.requests.append(SimpleNamespace(req_id=rid, token_ids=toks))
        return m

    sched._build_connector_meta_inner = lambda out: inner(
        out, [("ra", list(range(300)) + [1] * 300),
              ("rb", list(range(300)) + [2] * 300)])
    meta1 = sched.build_connector_meta(None)
    assert meta1.specagent_req_jobs == {"ra": "A", "rb": "B"}
    assert meta1.specagent_free_jobs == []

    sched.specagent_job_finished("A")
    sched.specagent_job_finished("A")  # deduplicated
    sched._build_connector_meta_inner = lambda out: inner(out, [])
    meta2 = sched.build_connector_meta(None)
    assert meta2.specagent_free_jobs == ["A"]
    assert "ra" not in sched._sa_req_job and "rb" in sched._sa_req_job
    meta3 = sched.build_connector_meta(None)
    assert meta3.specagent_free_jobs == []

    # Worker side.
    storage = FakeStorage({"a0", "a1"})
    worker = object.__new__(ad.LMCacheConnectorV1Impl)
    engine = SimpleNamespace(storage_manager=storage)
    worker._manager = SimpleNamespace(lmcache_engine=engine)
    joined = []
    worker._specagent_join_save_futures = lambda: joined.append(1)
    current = {"meta": meta1}
    worker._parent = SimpleNamespace(
        _get_connector_metadata=lambda: current["meta"])
    m = ad._specagent_worker_sync_jobs(worker)
    ad._specagent_worker_free_jobs(worker, m)
    reg = jf.registry_for(engine)
    assert reg.common_prefix == 300 and joined == []
    reg.register("ra", ["a0", "a1"], [256, 600], [1, 1])
    current["meta"] = meta2
    ad._specagent_worker_free_jobs(worker, ad._specagent_worker_sync_jobs(worker))
    assert joined == [1]
    assert storage.keys == {"a0"}
