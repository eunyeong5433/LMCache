# SPDX-License-Identifier: Apache-2.0
"""Drop a finished agent task's CPU KV cache (all systems, both models).

A task's KV entries are never read again once its last request has finished,
yet LMCache keeps them until LRU eviction, and LRU can pick a live task that
has waited long instead.  The scheduler therefore marks a task finished when
its last request (``is_last_step``) releases its blocks, i.e. after that
turn's CPU save completed, and the worker removes the task's chunks here.

Every chunk a store publishes is recorded under the job of the storing
request.  On finish the job's chunks are removed from the CPU backend except
  * chunks another job also stored (identical prefix), and
  * chunks that lie inside the token prefix common to every job's first
    request (system prompt / tool schema): other jobs hit those chunks
    without storing them.
Counters live in ``builtins._specagent_cpu_kv`` so the scheduler's stats
endpoint can export them for every system.
"""

from __future__ import annotations

import builtins
from typing import Any, Iterable, Optional

_STATS_KEYS = (
    "evicted",  # keys LRU-evicted by CPU allocation (any nonzero = pressure)
    "alloc_failed",  # CPU allocations that failed even after eviction
    "freed_jobs",
    "freed_keys",
    "freed_bytes",
    "kept_shared_keys",  # stored by more than one job
    "kept_common_keys",  # inside the common first-request prefix
    "remove_missing",  # registered but already gone at free time
    "late_keys",  # registered after the job was freed (left in place)
    "live_bytes",  # bytes of registered, not yet freed keys
    "peak_live_bytes",
    "common_prefix_tokens",
)


def stats() -> dict:
    st = vars(builtins).get("_specagent_cpu_kv")
    if st is None:
        st = {k: 0 for k in _STATS_KEYS}
        builtins._specagent_cpu_kv = st
    return st


def note_cpu_alloc(evicted: int, failed: bool) -> None:
    """Called by LocalCPUBackend.allocate / batched_allocate on every call."""
    if not evicted and not failed:
        return
    st = stats()
    st["evicted"] += int(evicted)
    st["alloc_failed"] += int(bool(failed))


class JobKeyRegistry:
    def __init__(self) -> None:
        self.req_job: dict[str, str] = {}
        self.job_reqs: dict[str, set[str]] = {}
        # job -> {key: (end_token, nbytes)}
        self.job_keys: dict[str, dict[Any, tuple[int, int]]] = {}
        self.owner: dict[Any, str] = {}
        self.shared: set[Any] = set()
        self.freed: set[str] = set()
        self._ref_tokens: Optional[list[int]] = None
        self.common_prefix: Optional[int] = None
        self._first_seen: set[str] = set()

    # -- request -> job mapping, from the scheduler's connector metadata --
    def note_requests(
        self, req_jobs: dict[str, str], first_tokens: dict[str, list[int]]
    ) -> None:
        for req_id, job in req_jobs.items():
            if not job:
                continue
            self.req_job[req_id] = job
            self.job_reqs.setdefault(job, set()).add(req_id)
            tokens = first_tokens.get(req_id)
            if job not in self._first_seen and tokens:
                self._first_seen.add(job)
                self._update_common_prefix(tokens)

    def _update_common_prefix(self, tokens: list[int]) -> None:
        if self._ref_tokens is None:
            self._ref_tokens = list(tokens)
            self.common_prefix = len(tokens)
        else:
            ref = self._ref_tokens
            n = min(len(ref), len(tokens), int(self.common_prefix or 0))
            i = 0
            while i < n and ref[i] == tokens[i]:
                i += 1
            self.common_prefix = i
        stats()["common_prefix_tokens"] = int(self.common_prefix or 0)

    def job_of(self, req_id: Optional[str]) -> Optional[str]:
        if req_id is None:
            return None
        return self.req_job.get(str(req_id))

    # -- store side --
    def register(
        self,
        req_id: Optional[str],
        keys: Iterable[Any],
        ends: Iterable[int],
        nbytes: Iterable[int],
    ) -> None:
        job = self.job_of(req_id)
        if job is None:
            return
        st = stats()
        if job in self.freed:
            st["late_keys"] += sum(1 for _ in keys)
            return
        mine = self.job_keys.setdefault(job, {})
        for key, end, size in zip(keys, ends, nbytes, strict=False):
            if key in self.shared:
                continue
            owner = self.owner.get(key)
            if owner is None:
                self.owner[key] = job
                mine[key] = (int(end), int(size))
                st["live_bytes"] += int(size)
                if st["live_bytes"] > st["peak_live_bytes"]:
                    st["peak_live_bytes"] = st["live_bytes"]
            elif owner != job:
                self.shared.add(key)
                st["kept_shared_keys"] += 1
                prev = self.job_keys.get(owner, {}).pop(key, None)
                if prev is not None:
                    st["live_bytes"] -= prev[1]
                del self.owner[key]

    # -- finish side --
    def free(self, job: str, storage_manager: Any) -> int:
        if job in self.freed:
            return 0
        self.freed.add(job)
        st = stats()
        keys = self.job_keys.pop(job, {})
        common = int(self.common_prefix or 0)
        remove: list[Any] = []
        removed_bytes = 0
        for key, (end, size) in keys.items():
            self.owner.pop(key, None)
            st["live_bytes"] -= size
            if end <= common:
                st["kept_common_keys"] += 1
                continue
            remove.append(key)
            removed_bytes += size
        n = 0
        if remove:
            n = int(
                storage_manager.batched_remove(
                    remove, locations=["LocalCPUBackend"]
                )
                or 0
            )
        st["freed_jobs"] += 1
        st["freed_keys"] += n
        st["freed_bytes"] += removed_bytes
        st["remove_missing"] += max(0, len(remove) - n)
        for req_id in self.job_reqs.pop(job, ()):
            self.req_job.pop(req_id, None)
        return n


def registry_for(engine: Any) -> JobKeyRegistry:
    reg = getattr(engine, "_specagent_job_registry", None)
    if reg is None:
        reg = JobKeyRegistry()
        engine._specagent_job_registry = reg
    return reg
