# SPDX-License-Identifier: Apache-2.0
"""Per-request VERIFY plan cache: reuse, lease accounting and invalidation."""

# Standard
import os

# Third Party
import pytest

# First Party
from lmcache.v1.cache_engine import (
    LMCacheEngine,
    SpecAgentPreparedLayerwiseBatch,
    SpecAgentPreparedLayerwiseSegment,
)


class FakeOwner:
    def __init__(self, name):
        self.name, self.refs, self.valid = name, 1, True

    def ref_count_up(self):
        self.refs += 1

    def ref_count_down(self):
        self.refs -= 1

    def is_valid(self):
        return self.valid


@pytest.fixture
def engine(monkeypatch):
    eng = LMCacheEngine.__new__(LMCacheEngine)
    owners = {}
    calls = {"n": 0}

    def prepare(plans):
        calls["n"] += 1
        (plan,) = plans
        objs = []
        for s, e, k in plan:
            o = owners.setdefault(k, FakeOwner(k))
            o.ref_count_up()  # acquisition lease, like batched_get_blocking
            objs.append(o)
        seg = SpecAgentPreparedLayerwiseSegment(
            starts=[s for s, _, _ in plan], ends=[e for _, e, _ in plan],
            keys_by_chunk=[[k] for _, _, k in plan], location="cpu",
            memory_objs_by_layer=[["view"] * len(plan)], source_owners=objs)
        return SpecAgentPreparedLayerwiseBatch([seg])

    monkeypatch.setattr(eng, "prepare_sparse_layerwise_plans", prepare)
    monkeypatch.setenv("SPECAGENT_VERIFY_PLAN_CACHE_MAX_REQS", "2")
    return eng, owners, calls


def consume(batch):
    seg = batch.claim(0)
    seg.release_source_owners()  # consumer releases at GPU completion
    return seg


def test_hit_reuses_and_balances_leases(engine):
    eng, owners, calls = engine
    plan = [(0, 256, "a"), (256, 512, "b")]
    consume(eng._specagent_prepare_with_plan_cache("r1", plan))
    assert calls["n"] == 1 and owners["a"].refs == 2  # backend + cache
    seg = eng._specagent_prepare_with_plan_cache("r1", plan).claim(0)
    assert calls["n"] == 1  # reused, no fresh preparation
    assert owners["a"].refs == 3  # backend + cache + this round
    seg.release_source_owners()
    assert owners["a"].refs == 2


def test_changed_plan_replaces_and_releases(engine):
    eng, owners, calls = engine
    consume(eng._specagent_prepare_with_plan_cache("r1", [(0, 256, "a")]))
    consume(eng._specagent_prepare_with_plan_cache("r1", [(0, 256, "c")]))
    assert calls["n"] == 2 and owners["a"].refs == 1 and owners["c"].refs == 2


def test_lru_eviction_releases(engine):
    eng, owners, _ = engine
    for r, k in (("r1", "a"), ("r2", "b"), ("r3", "c")):
        consume(eng._specagent_prepare_with_plan_cache(r, [(0, 256, k)]))
    assert owners["a"].refs == 1  # r1 evicted (max 2 requests)
    assert owners["b"].refs == 2 and owners["c"].refs == 2


def test_invalid_owner_forces_fresh(engine):
    eng, owners, calls = engine
    consume(eng._specagent_prepare_with_plan_cache("r1", [(0, 256, "a")]))
    owners["a"].valid = False
    consume(eng._specagent_prepare_with_plan_cache("r1", [(0, 256, "a")]))
    assert calls["n"] == 2


def test_check_mode_passes_on_identical(engine, monkeypatch):
    eng, owners, calls = engine
    monkeypatch.setenv("SPECAGENT_VERIFY_PLAN_CACHE_CHECK", "1")
    plan = [(0, 256, "a")]
    consume(eng._specagent_prepare_with_plan_cache("r1", plan))
    consume(eng._specagent_prepare_with_plan_cache("r1", plan))
    assert calls["n"] == 2 and owners["a"].refs == 2


class FakeTokenDB:
    chunk_size = 4

    def __init__(self):
        self.calls = 0

    def process_tokens(self, tokens, request_configs=None, **_):
        self.calls += 1
        for s in range(0, len(tokens), 4):
            yield (s, min(s + 4, len(tokens)), f"k{s}")


@pytest.fixture
def plan_engine(monkeypatch):
    eng = LMCacheEngine.__new__(LMCacheEngine)
    eng.token_database = FakeTokenDB()
    monkeypatch.setenv("SPECAGENT_VERIFY_CHUNKPLAN_CACHE", "1")
    return eng


def test_chunk_plan_reused_for_same_request_and_segments(plan_engine, monkeypatch):
    # plan_sparse_layerwise_chunks only yields CacheEngineKey instances in prod;
    # relax the isinstance assertion for the fake keys.
    monkeypatch.setattr("lmcache.v1.cache_engine.CacheEngineKey", str)
    toks = list(range(16))
    p1 = plan_engine.plan_sparse_layerwise_chunks(toks, [(4, 12)], None, memo_id="r1")
    p2 = plan_engine.plan_sparse_layerwise_chunks(toks, [(4, 12)], None, memo_id="r1")
    assert p1 == p2 and plan_engine.token_database.calls == 1
    plan_engine.plan_sparse_layerwise_chunks(toks, [(0, 12)], None, memo_id="r1")
    plan_engine.plan_sparse_layerwise_chunks(toks, [(4, 12)], None, memo_id="r2")
    assert plan_engine.token_database.calls == 3


def test_chunk_plan_check_mode(plan_engine, monkeypatch):
    monkeypatch.setattr("lmcache.v1.cache_engine.CacheEngineKey", str)
    monkeypatch.setenv("SPECAGENT_VERIFY_PLAN_CACHE_CHECK", "1")
    toks = list(range(16))
    plan_engine.plan_sparse_layerwise_chunks(toks, [(4, 12)], None, memo_id="r1")
    plan_engine.plan_sparse_layerwise_chunks(toks, [(4, 12)], None, memo_id="r1")
    assert plan_engine.token_database.calls == 2


def _cyclic(eng, rounds=6):
    for _ in range(rounds):
        for r, k in (("r1", "a"), ("r2", "b"), ("r3", "c")):
            consume(eng._specagent_prepare_with_plan_cache(r, [(0, 256, k)]))


def test_lru_thrashes_on_cyclic_access(engine):
    eng, owners, calls = engine  # MAX_REQS=2, 3 requests in a cycle
    _cyclic(eng)
    assert calls["n"] == 18  # every call misses


def test_keep_admission_serves_cyclic_access(engine, monkeypatch):
    eng, owners, calls = engine
    monkeypatch.setenv("SPECAGENT_VERIFY_PLAN_CACHE_ADMIT", "keep")
    _cyclic(eng)
    # r1, r2 stay resident; only r3 is prepared every round (+2 cold misses)
    assert calls["n"] == 2 + 6
    assert owners["a"].refs == 2 and owners["b"].refs == 2  # backend + cache
    assert owners["c"].refs == 1  # never held by the cache


def test_keep_admission_evicts_stale(engine, monkeypatch):
    eng, owners, calls = engine
    monkeypatch.setenv("SPECAGENT_VERIFY_PLAN_CACHE_ADMIT", "keep")
    monkeypatch.setenv("SPECAGENT_VERIFY_PLAN_CACHE_STALE_CALLS", "3")
    for r, k in (("r1", "a"), ("r2", "b")):
        consume(eng._specagent_prepare_with_plan_cache(r, [(0, 256, k)]))
    for _ in range(4):  # r1 keeps hitting; r2 goes stale
        consume(eng._specagent_prepare_with_plan_cache("r1", [(0, 256, "a")]))
    consume(eng._specagent_prepare_with_plan_cache("r3", [(0, 256, "c")]))
    assert owners["b"].refs == 1 and owners["c"].refs == 2 and owners["a"].refs == 2
