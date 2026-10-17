# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import defaultdict
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Tuple,
    Union,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.health_monitor.base import HealthMonitor

# Standard
import asyncio
import builtins
import gc
import multiprocessing
import os
import threading
import time

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.observability import LMCacheStatsLogger, LMCStatsMonitor
from lmcache.usage_context import InitializeUsageContext
from lmcache.v1.specagent_layer_group import LayerGroup, chunk_key
from lmcache.v1.specagent_bulk_view import (
    BorrowedBulkLayerView, acquire_bulk_layer_views, borrow_bulk_layer_views,
    merge_native_bulk_prefix,
)
from lmcache.utils import (
    CacheEngineKey,
    CacheStoreEvent,
    _lmcache_nvtx_annotate,
    compress_slot_mapping,
    convert_tokens_to_list,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventManager, EventStatus, EventType
from lmcache.v1.gpu_connector.gpu_connectors import GPUConnectorInterface
from lmcache.v1.gpu_connector.utils import assert_layerwise_gpu_connector
from lmcache.v1.memory_management import CuFileMemoryAllocator  # noqa: E501
from lmcache.v1.memory_management import (  # noqa: E501
    MemoryAllocatorInterface,
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    MixedMemoryAllocator,
    PagedTensorMemoryAllocator,
    TensorMemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.pin_monitor import PinMonitor
from lmcache.v1.storage_backend.storage_manager import StorageManager
from lmcache.v1.system_detection import NUMADetector, NUMAMapping
from lmcache.v1.token_database import (
    ChunkedTokenDatabase,
    SegmentTokenDatabase,
    TokenDatabase,
)

logger = init_logger(__name__)

_SPECAGENT_LIVE_PROBE_LOG_LIMIT = 20


class _LazyLayerKeys:
    """Per-layer keys of one chunk, split only if a consumer iterates them."""

    __slots__ = ("key", "num_layers", "_split")

    def __init__(self, key, num_layers: int) -> None:
        self.key = key
        self.num_layers = int(num_layers)
        self._split = None

    def layer0(self):
        return self.key.split_layers(1)[0]

    def _keys(self):
        if self._split is None:
            self._split = self.key.split_layers(self.num_layers)
        return self._split

    def __len__(self) -> int:
        return self.num_layers

    def __iter__(self):
        return iter(self._keys())

    def __getitem__(self, index):
        return self._keys()[index]


@dataclass
class SpecAgentPreparedLayerwiseSegment:
    """CPU-only sparse retrieval plan whose MemoryObj refs are leased."""

    starts: list[int]
    ends: list[int]
    keys_by_chunk: list[list[CacheEngineKey]]
    location: str
    memory_objs_by_layer: list[list[MemoryObj | BorrowedBulkLayerView]]
    source_owners: Optional[list[MemoryObj]] = None
    # (layer-0 source tensors, per-source layer byte deltas) precomputed for
    # the GPU connector's deferred layer-0 collection; None -> derive there.
    layer0_sources: Optional[tuple] = None

    def release_source_owners(self) -> None:
        """Release chunk leases still held here, including abandoned claims."""
        owners = self.source_owners
        self.source_owners = [] if owners is not None else None
        for owner in owners or []:
            owner.ref_count_down()

    def __del__(self) -> None:
        try:
            self.release_source_owners()
        except Exception:
            pass


class SpecAgentPreparedLayerwiseBatch:
    """Own prepared segment leases until each consumer claims them.

    ``batched_get_blocking`` increments MemoryObj reference counts.  A segment
    transfers those references to ``retrieve_layer`` exactly once via
    :meth:`claim`; any unclaimed segment is released by the batch owner.
    """

    def __init__(
        self, segments: list[SpecAgentPreparedLayerwiseSegment]
    ) -> None:
        self._segments: list[Optional[SpecAgentPreparedLayerwiseSegment]] = (
            list(segments)
        )
        self._lock = threading.Lock()

    def claim(self, index: int) -> SpecAgentPreparedLayerwiseSegment:
        with self._lock:
            segment = self._segments[index]
            if segment is None:
                raise RuntimeError(
                    f"SpecAgent prepared segment {index} was already claimed"
                )
            self._segments[index] = None
            return segment

    def release_unclaimed(self) -> None:
        with self._lock:
            segments = [seg for seg in self._segments if seg is not None]
            self._segments = [None] * len(self._segments)
        for segment in segments:
            if segment.source_owners is not None:
                segment.release_source_owners()
            else:
                for mem_objs_layer in segment.memory_objs_by_layer:
                    for mem_obj in mem_objs_layer:
                        mem_obj.ref_count_down()

    def __del__(self) -> None:
        try:
            self.release_unclaimed()
        except Exception:
            # Destructors must remain best-effort during interpreter teardown.
            pass


class SpecAgentProgressiveLayerwiseSegment:
    """Layerwise plan whose MemoryObj leases are published incrementally."""

    def __init__(
        self,
        *,
        starts: list[int],
        ends: list[int],
        keys_by_chunk: list[list[CacheEngineKey]],
        location: str,
        num_layers: int,
    ) -> None:
        self.starts = starts
        self.ends = ends
        self.keys_by_chunk = keys_by_chunk
        self.location = location
        self._layers: list[Optional[list[MemoryObj]]] = [None] * num_layers
        self._condition = threading.Condition()
        self._error: Optional[BaseException] = None
        self._released = False

    def publish(self, layer_id: int, memory_objs: list[MemoryObj]) -> None:
        release_now = False
        with self._condition:
            if self._released:
                release_now = True
            else:
                self._layers[layer_id] = memory_objs
                self._condition.notify_all()
        if release_now:
            for mem_obj in memory_objs:
                mem_obj.ref_count_down()

    def fail(self, error: BaseException) -> None:
        with self._condition:
            self._error = error
            self._condition.notify_all()

    def take_layer(self, layer_id: int) -> list[MemoryObj]:
        with self._condition:
            self._condition.wait_for(
                lambda: (
                    self._layers[layer_id] is not None
                    or self._error is not None
                    or self._released
                )
            )
            if self._error is not None:
                raise self._error
            if self._released:
                raise RuntimeError("SpecAgent progressive plan was released")
            memory_objs = self._layers[layer_id]
            assert memory_objs is not None
            self._layers[layer_id] = None
            return memory_objs

    def release_pending(self) -> None:
        with self._condition:
            self._released = True
            pending = [row for row in self._layers if row is not None]
            self._layers = [None] * len(self._layers)
            self._condition.notify_all()
        for row in pending:
            for mem_obj in row:
                mem_obj.ref_count_down()

    def __del__(self) -> None:
        try:
            self.release_pending()
        except Exception:
            pass


class SpecAgentProgressiveLayerwiseBatch:
    """Shared handle published before all layer MemoryObjs are resolved."""

    def __init__(self, num_segments: int) -> None:
        self._segments: Optional[
            list[Optional[SpecAgentProgressiveLayerwiseSegment]]
        ] = None
        self._num_segments = num_segments
        self._condition = threading.Condition()
        self._error: Optional[BaseException] = None

    def publish_segments(
        self, segments: list[SpecAgentProgressiveLayerwiseSegment]
    ) -> None:
        if len(segments) != self._num_segments:
            raise ValueError("progressive segment count mismatch")
        with self._condition:
            self._segments = list(segments)
            self._condition.notify_all()

    def publish_unavailable(self) -> None:
        with self._condition:
            self._segments = []
            self._condition.notify_all()

    def fail(self, error: BaseException) -> None:
        with self._condition:
            self._error = error
            segments = list(self._segments or [])
            self._condition.notify_all()
        for segment in segments:
            if segment is not None:
                segment.fail(error)

    def claim(
        self, index: int
    ) -> Optional[SpecAgentProgressiveLayerwiseSegment]:
        with self._condition:
            self._condition.wait_for(
                lambda: self._segments is not None or self._error is not None
            )
            if self._error is not None:
                raise self._error
            assert self._segments is not None
            if not self._segments:
                return None
            segment = self._segments[index]
            if segment is None:
                raise RuntimeError(
                    f"SpecAgent progressive segment {index} was already claimed"
                )
            self._segments[index] = None
            return segment

    def release_unclaimed(self) -> None:
        with self._condition:
            segments = list(self._segments or [])
            if self._segments is not None:
                self._segments = [None] * len(self._segments)
        for segment in segments:
            if segment is not None:
                segment.release_pending()

    def __del__(self) -> None:
        try:
            self.release_unclaimed()
        except Exception:
            pass

try:
    # vLLM exposes NVTX ranges only when profiling asks for them.  Keep a
    # no-op fallback so standalone LMCache imports do not depend on vLLM.
    from vllm.v1.utils import record_function_or_nullcontext as _lmc_record
except Exception:  # pragma: no cover - vLLM may be absent in LMCache tests.
    def _lmc_record(_name: str):
        return nullcontext()


def _specagent_record_actual_cpu_store(
    tokens: Union[torch.Tensor, list[int]],
    start: int,
    end: int,
    block_size: int,
    *,
    prefix_snapshot: Optional[tuple[int, ...]] = None,
    prefix_hashes: Optional[dict[int, int]] = None,
) -> None:
    if os.environ.get("SPECAGENT_DISABLE_CPU_STORE_PROOF"):
        return

    chunk_store = getattr(builtins, "_specagent_cpu_saved_chunks", None)
    if chunk_store is None:
        chunk_store = set()
        setattr(builtins, "_specagent_cpu_saved_chunks", chunk_store)

    token_hash = (
        prefix_hashes.get(end)
        if prefix_snapshot is not None and prefix_hashes is not None else None
    )
    prefix_tokens = ()
    if (token_hash is None or os.environ.get("SPECAGENT_HASH_DEBUG")
            or os.environ.get("SPECAGENT_LEGACY_PRESTORE_PROOF")):
        if prefix_snapshot is not None:
            prefix_tokens = prefix_snapshot[:end]
        elif isinstance(tokens, torch.Tensor):
            prefix_tokens = tuple(int(t) for t in tokens[:end].detach().cpu().tolist())
        else:
            prefix_tokens = tuple(int(t) for t in tokens[:end])
    if token_hash is None:
        token_hash = hash(prefix_tokens)
    if prefix_hashes is not None:
        prefix_hashes[end] = token_hash
    chunk_store.add((end, token_hash))
    if (
        os.environ.get("SPECAGENT_TAIL_SAVE_DEBUG")
        and end % 256 != 0
    ):
        logger.error(
            "[SpecAgent][OBS-TAIL-PROOF] end=%d start=%d hash=%d",
            end,
            start,
            token_hash,
        )
    builtins._sa_record_n = getattr(builtins, "_sa_record_n", 0) + 1
    if os.environ.get("SPECAGENT_HASH_DEBUG") and end <= 512:
        import logging as _lg
        _lg.getLogger("lmcache").info(
            "[SpecAgent][HASH-STORE] end=%d hash=%d len=%d head=%s tail=%s",
            end, token_hash, len(prefix_tokens),
            prefix_tokens[:3], prefix_tokens[-3:],
        )

    if not os.environ.get("SPECAGENT_LEGACY_PRESTORE_PROOF"):
        return

    block_store = getattr(builtins, "_specagent_cpu_saved_blocks", None)
    if block_store is None:
        block_store = set()
        setattr(builtins, "_specagent_cpu_saved_blocks", block_store)
    for tok_end in range(
        ((start + block_size - 1) // block_size) * block_size,
        end + 1,
        block_size,
    ):
        if tok_end <= start:
            continue
        block_prefix = prefix_tokens[:tok_end]
        block_store.add((tok_end // block_size - 1, hash(tuple(block_prefix))))


def _specagent_prepare_cpu_store_proofs(
    tokens: Union[torch.Tensor, list[int]],
    ends: list[int],
) -> tuple[Optional[tuple[int, ...]], dict[int, int], list]:
    """Compute immutable prefix hashes without publishing storage evidence."""
    if not ends:
        return None, {}, []
    proof_prefix = None
    if not os.environ.get("SPECAGENT_DISABLE_CPU_STORE_PROOF"):
        proof_tokens = tokens[:max(ends)]
        if isinstance(proof_tokens, torch.Tensor):
            proof_tokens = proof_tokens.detach().cpu().tolist()
        proof_prefix = tuple(proof_tokens)
    proof_hashes = {}
    snapshots = getattr(
        builtins, "_specagent_cpu_store_hash_snapshots", ()
    )
    retained_snapshots = []
    if proof_prefix is not None:
        # Cache hash computations only, never save validity.
        # Coalesce an extending prefix with its ancestors:
        # repeated stores for one job must not evict the
        # other active jobs' reusable hash computations.
        for previous_prefix, previous_hashes in snapshots:
            common = min(len(previous_prefix), len(proof_prefix))
            matches = previous_prefix[:common] == proof_prefix[:common]
            if matches:
                proof_hashes.update(
                    (end, value) for end, value in previous_hashes.items()
                    if end <= common
                )
            if not matches or len(previous_prefix) > len(proof_prefix):
                retained_snapshots.append((previous_prefix, previous_hashes))
    if proof_prefix is not None:
        for end in ends:
            if end not in proof_hashes:
                proof_hashes[end] = hash(proof_prefix[:end])
    return proof_prefix, proof_hashes, retained_snapshots


def _specagent_publish_cpu_store_proofs(
    tokens: Union[torch.Tensor, list[int]],
    starts: list[int],
    ends: list[int],
    specagent_block_size: int,
    *,
    prepared: Optional[tuple] = None,
) -> None:
    """Publish exact prefix proofs only after all layers are stored on CPU."""
    if not ends:
        return
    proof_prefix, proof_hashes, retained_snapshots = (
        _specagent_prepare_cpu_store_proofs(tokens, ends)
        if prepared is None else prepared
    )
    for start, end in zip(starts, ends, strict=True):
        _specagent_record_actual_cpu_store(
            tokens, start, end, specagent_block_size,
            prefix_snapshot=proof_prefix,
            prefix_hashes=proof_hashes,
        )
    if proof_prefix is not None:
        builtins._specagent_cpu_store_hash_snapshots = (
            retained_snapshots[-7:] + [(proof_prefix, proof_hashes)]
        )


def _specagent_record_actual_kv_fingerprint(
    tokens: Union[torch.Tensor, list[int]],
    memory_obj: MemoryObj,
    start: int,
    end: int,
    block_size: int,
    req_id: str,
    layer_id: int = 0,
    cache_key: Optional[CacheEngineKey] = None,
) -> None:
    if not bool(__import__("os").environ.get("SPECAGENT_KV_FINGERPRINT")):
        return
    if memory_obj.tensor is None:
        return
    tensor = memory_obj.tensor
    if tensor.dim() >= 3 and tensor.shape[0] == 1 and tensor.shape[1] == 2:
        tensor = tensor[0]
    elif tensor.dim() == 3 and tensor.shape[1] == 2:
        tensor = tensor.permute(1, 0, 2).contiguous()
    if tensor.dim() < 2 or tensor.shape[0] != 2:
        logger.warning(
            "[SpecAgent][KV-FP-SKIP] req=%s reason=unexpected_memory_shape "
            "shape=%s",
            req_id,
            tuple(memory_obj.tensor.shape),
        )
        return
    fp_store = getattr(builtins, "_specagent_kv_fingerprints", None)
    if fp_store is None:
        fp_store = {}
        setattr(builtins, "_specagent_kv_fingerprints", fp_store)

    for tok_start in range(
        (start // block_size) * block_size,
        end,
        block_size,
    ):
        tok_end = min(tok_start + block_size, end)
        if tok_start < start or tok_end <= tok_start:
            continue
        target_logicals_raw = __import__("os").environ.get(
            "SPECAGENT_KV_FINGERPRINT_TARGET_LOGICALS", ""
        ).strip()
        if target_logicals_raw:
            target_logicals: set[int] = set()
            for part in target_logicals_raw.split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    target_logicals.add(int(part))
                except ValueError:
                    continue
            if target_logicals and tok_start // block_size not in target_logicals:
                continue
        if isinstance(tokens, torch.Tensor):
            prefix_tokens = [
                int(t) for t in tokens[:tok_end].detach().cpu().tolist()
            ]
        else:
            prefix_tokens = [int(t) for t in tokens[:tok_end]]
        rel_start = tok_start - start
        rel_end = tok_end - start
        kv = tensor[:, rel_start:rel_end].float()
        fp_value = (
            tok_end - tok_start,
            float(kv[0].sum().item()),
            float(kv[1].sum().item()),
            float((kv[0] * kv[0]).sum().item()),
            float((kv[1] * kv[1]).sum().item()),
        )
        # Use LMCache's authoritative chunk key when available. Logical block
        # indices alone collide across unrelated requests and are only useful
        # as human-readable context in the log.
        key_hash = (
            cache_key.chunk_hash
            if cache_key is not None
            else hash(tuple(prefix_tokens))
        )
        fp_key = (
            int(layer_id),
            key_hash,
            tok_start - start,
            tok_end - tok_start,
        )
        legacy_prefix_key = (
            int(layer_id),
            tok_start // block_size,
            hash(tuple(prefix_tokens)),
        )
        old_versions = fp_store.get(fp_key)
        if old_versions is None:
            old_versions = []
        elif isinstance(old_versions, tuple):
            old_versions = [old_versions]
        if old_versions:
            deltas = []
            for old_fp in old_versions:
                sum_delta = max(
                    abs(float(old_fp[1]) - fp_value[1]),
                    abs(float(old_fp[2]) - fp_value[2]),
                )
                norm_delta = max(
                    abs(float(old_fp[3]) - fp_value[3]),
                    abs(float(old_fp[4]) - fp_value[4]),
                )
                deltas.append((sum_delta, norm_delta))
            best_sum_delta, best_norm_delta = min(
                deltas, key=lambda item: (item[0], item[1])
            )
            if best_sum_delta > 1e-3 or best_norm_delta > 1e-2:
                logger.warning(
                    "[SpecAgent][KV-FP-VERSION] req=%s layer=%d logical=%d "
                    "token_range=[%d,%d) chunk_start=%d key=%s prev_versions=%d "
                    "best_sum_delta=%.6g best_norm_delta=%.6g source=actual_store",
                    req_id,
                    layer_id,
                    tok_start // block_size,
                    tok_start,
                    tok_end,
                    start,
                    str(key_hash),
                    len(old_versions),
                    best_sum_delta,
                    best_norm_delta,
                )
        old_versions.append(fp_value)
        if len(old_versions) > 8:
            old_versions = old_versions[-8:]
        fp_store[fp_key] = old_versions
        # Keep the older logical-block key as a debug alias.  The runtime
        # correctness checker in vLLM indexes by logical block and prefix hash,
        # while LMCache's authoritative store key is chunk based.
        legacy_versions = fp_store.get(legacy_prefix_key)
        if legacy_versions is None:
            legacy_versions = []
        elif isinstance(legacy_versions, tuple):
            legacy_versions = [legacy_versions]
        legacy_versions.append(fp_value)
        if len(legacy_versions) > 8:
            legacy_versions = legacy_versions[-8:]
        fp_store[legacy_prefix_key] = legacy_versions

# Type aliases for processed chunks
# (cache_key, memory_obj, start_index, end_index)
ProcessedChunk = Tuple[CacheEngineKey, MemoryObj, int, int]
# (list of processed chunks, total kv size)
ProcessTokensInternalResult = Tuple[List[ProcessedChunk], int]


class CacheEngineEndSignal:
    pass


class LMCacheEngine:
    """The main class for the cache engine.

    When storing the KV caches into the cache engine, it takes GPU KV
    caches from the serving engine and convert them into MemoryObjs that
    resides in the CPU. The MemoryObjs are then being stored into the
    StorageBackends in an asynchronous manner.

    When retrieving the KV caches from the cache engine, it fetches the
    MemoryObjs from the StorageBackends and convert them into GPU KV caches
    by GPUConnectors specialized for the serving engine.

    It also supports prefetching the KV caches from the StorageBackends.
    It relies on the StorageBackends to manage the requests of prefetching
    and real retrieval and avoid the conflicts.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        token_database: TokenDatabase,
        gpu_connector: Optional[GPUConnectorInterface],
        broadcast_fn: Callable[[torch.Tensor, int], None],
        broadcast_object_fn: Callable[[Any, int], Any],
    ):
        logger.info(f"Creating LMCacheEngine with config: {config}")
        self.config = config
        self.metadata = metadata
        self.token_database = token_database
        self.gpu_connector = gpu_connector
        self.broadcast_fn = broadcast_fn
        self.broadcast_object_fn = broadcast_object_fn
        # save_only_first_rank only works when use mla
        self.save_only_first_rank = (
            self.config.get_extra_config_value("save_only_first_rank", metadata.use_mla)
            and metadata.use_mla
        )

        if self.save_only_first_rank and self.gpu_connector is not None:
            self.broadcast_stream = (
                self.gpu_connector.load_stream
                if hasattr(self.gpu_connector, "load_stream")
                else torch.cuda.Stream()
            )

        self.enable_controller = config.enable_controller

        # NOTE: Unix systems use fork by default
        multiprocessing.set_start_method("spawn", force=True)

        # avoid circular import
        # First Party
        from lmcache.v1.cache_controller import LMCacheWorker

        self.lmcache_worker: Optional[LMCacheWorker] = None
        lmcache_worker_ids = config.get_lmcache_worker_ids(
            metadata.use_mla, metadata.world_size
        )
        # lmcache_worker_ids is empty means start on all workers
        if (
            self.enable_controller
            and self.metadata.role != "scheduler"
            and (not lmcache_worker_ids or metadata.worker_id in lmcache_worker_ids)
        ):
            self.lmcache_worker = LMCacheWorker(config, metadata, self)
        else:
            self.lmcache_worker = None
            logger.info(
                "LMCacheWorker is not initialized (related configs: "
                "enable_controller: %s, role: %s, worker_id: %s, worker_ids: %s).",
                self.enable_controller,
                self.metadata.role,
                self.metadata.worker_id,
                lmcache_worker_ids,
            )

        self.async_loading = config.enable_async_loading
        self.event_manager = EventManager()

        self.use_layerwise = config.use_layerwise
        self.native_bulk_spec = os.environ.get("SPECAGENT_NATIVE_BULK_SPEC") == "1"
        if self.native_bulk_spec and (
            self.use_layerwise or metadata.use_mla or config.enable_blending
            or config.enable_async_loading or os.environ.get("SPECAGENT_PROGRESSIVE_CPU_PREPARE")
        ):
            raise ValueError("Native bulk SPEC requires synchronous, non-MLA bulk storage and full-plan fetch")
        self.grouped_layers = (
            self.use_layerwise and os.environ.get("SPECAGENT_CPU_LAYER_GROUP") == "1"
        )
        if getattr(self, "grouped_layers", False) and (
            metadata.use_mla or config.enable_blending or config.enable_async_loading
            or os.environ.get("SPECAGENT_DIRECT_BULK_SAVE") != "1"
            or os.environ.get("SPECAGENT_KV_FINGERPRINT_PRESTORE")
        ):
            raise ValueError("Grouped CPU KV requires synchronous T2D direct bulk save")

        # TODO: support save_only_first_rank when use layerwise
        # if use_layerwise is True, all ranks will initialize the storage_manager
        # if save_only_first_rank is False, all ranks will initialize
        # the storage_manager
        # if save_only_first_rank is True, only the first rank and
        # lookup server workers will initialize the storage_manager
        self.storage_manager: Optional[StorageManager] = None

        # KV events
        self.kv_events_enabled = False
        self.kv_events_enabled = config.enable_kv_events
        if self.kv_events_enabled:
            self.kv_events: List[CacheStoreEvent] = []
            logger.info("KV events are enabled.")
        else:
            logger.info("KV events are disabled.")

        # HACK: remove this in the future
        # NOTE (Jiayi): This is currently used to support
        # dropping the kv cache from the buffer in PD backend
        # at decoder.
        self.remove_after_retrieve = config.enable_pd and config.pd_role == "receiver"

        # asymmetric store/retrieve location can be specified
        # this is typically used (but not limited) in PD system
        self.store_location = config.store_location
        self.retrieve_locations = config.retrieve_locations

        self.num_layers = metadata.kv_shape[0]
        self.fmt = None
        if self.use_layerwise:
            if metadata.use_mla:
                self.fmt = MemoryFormat.KV_MLA_FMT
            elif config.enable_blending:
                self.fmt = MemoryFormat.KV_2TD
            else:
                self.fmt = MemoryFormat.KV_T2D
        if metadata.use_mla:
            self.fmt = MemoryFormat.KV_MLA_FMT

        # NOTE(ApostaC): we haven't support lookup-cache yet
        self.lookup_cache: dict[CacheEngineKey, Any] = {}

        # lookup_id -> {location -> [pinned keys]}
        self.lookup_pins: dict[str, dict[str, list]] = defaultdict(
            lambda: defaultdict(list)
        )

        InitializeUsageContext(config, metadata)
        self.stats_monitor = LMCStatsMonitor.GetOrCreate()
        # Initialize PinMonitor singleton with config
        PinMonitor.GetOrCreate(config)

        self.post_inited = False

        # Flag to control KVCache Check logging (can be toggled via API)
        self.kvcache_check_log_enabled = False

        gc.collect()
        if not config.py_enable_gc:
            gc.disable()

        # Health monitor reference (injected by LMCacheManager)
        self._health_monitor: Optional["HealthMonitor"] = None

        # Flag to indicate if initialization failed (irrecoverable error)
        self._init_failed = False

    def set_health_monitor(self, health_monitor: "HealthMonitor") -> None:
        """
        Set the health monitor reference.

        This is called by LMCacheManager after creating the HealthMonitor
        to inject the reference into the engine.

        Args:
            health_monitor: The HealthMonitor instance from LMCacheManager
        """
        self._health_monitor = health_monitor

    def is_healthy(self) -> bool:
        """
        Check if the LMCache system is healthy.

        This method returns False if:
        - Initialization failed (irrecoverable error)
        - HealthMonitor reports unhealthy

        If no health monitor is set and initialization succeeded,
        it returns True (assume healthy).

        Returns:
            bool: True if healthy, False otherwise
        """
        if self._init_failed:
            return False
        if self._health_monitor is not None:
            return self._health_monitor.is_healthy()
        return True

    def _get_req_id(self, kwargs: dict) -> str:
        """Extracts request ID from kwargs for logging."""
        return kwargs.get("req_id", "unspecified")

    def specagent_live_saved_tokens(
        self,
        tokens: Union[torch.Tensor, list[int]],
        upto_tokens: Optional[int] = None,
    ) -> int:
        """Return the live contiguous CPU-backed prefix length.

        LMCache layerwise retrieval stops at the first unavailable chunk.
        Therefore SpecAgent full restore can only safely advertise the
        contiguous prefix for which every layer of every chunk is currently
        present in the storage backend.
        """
        global _SPECAGENT_LIVE_PROBE_LOG_LIMIT
        debug_probe = bool(os.environ.get("SPECAGENT_LIVE_PROBE_DEBUG"))
        if self.storage_manager is None:
            if debug_probe and _SPECAGENT_LIVE_PROBE_LOG_LIMIT > 0:
                _SPECAGENT_LIVE_PROBE_LOG_LIMIT -= 1
                logger.info(
                    "[SpecAgent][CPU-LIVE-PROBE-LMCACHE] storage_manager=None "
                    "engine_id=%s worker_id=%s",
                    id(self),
                    getattr(self.metadata, "worker_id", None),
                )
            return 0
        limit = len(tokens) if upto_tokens is None else min(
            len(tokens), max(0, int(upto_tokens))
        )
        chunk_size = int(getattr(self.token_database, "chunk_size", 0) or 0)
        if chunk_size <= 0:
            return 0
        if limit <= 0:
            return 0

        live_tokens = 0
        probe_tokens = tokens[:limit]
        for start, end, key in self.token_database.process_tokens(
            tokens=probe_tokens,
            request_configs=None,
        ):
            assert isinstance(key, CacheEngineKey)
            keys_multi_layer = (
                [key] if not self.use_layerwise or getattr(self, "grouped_layers", False)
                else key.split_layers(self.num_layers)
            )
            layer_locations = [
                self.storage_manager.contains(layer_key, self.retrieve_locations)
                for layer_key in keys_multi_layer
            ]
            if debug_probe and _SPECAGENT_LIVE_PROBE_LOG_LIMIT > 0:
                _SPECAGENT_LIVE_PROBE_LOG_LIMIT -= 1
                present_layers = sum(1 for loc in layer_locations if loc)
                logger.info(
                    "[SpecAgent][CPU-LIVE-PROBE-LMCACHE] engine_id=%s "
                    "worker_id=%s chunk=[%d,%d) limit=%d chunk_size=%d "
                    "key_hash=%s present_layers=%d/%d first_location=%s "
                    "retrieve_locations=%s store_location=%s",
                    id(self),
                    getattr(self.metadata, "worker_id", None),
                    start,
                    end,
                    limit,
                    chunk_size,
                    str(key.chunk_hash),
                    present_layers,
                    self.num_layers,
                    str(layer_locations[0]) if layer_locations else None,
                    self.retrieve_locations,
                    self.store_location,
                )
            if not all(layer_locations):
                break
            live_tokens = int(end)
        return live_tokens

    def mark_init_failed(self, reason: str = "") -> None:
        """
        Mark the engine as having failed initialization.

        This is called by LMCacheManager when an irrecoverable error occurs
        during initialization or post_init. Once marked, is_healthy() will
        always return False, causing the system to fall back to recomputation.

        Args:
            reason: Optional reason string for logging
        """
        self._init_failed = True
        if reason:
            logger.error("LMCacheEngine marked as init failed: %s", reason)
        else:
            logger.error("LMCacheEngine marked as init failed")

    def post_init(self, **kwargs) -> None:
        if not self.post_inited:
            logger.info("Post initializing LMCacheEngine")
            lookup_server_worker_ids = self.config.get_lookup_server_worker_ids(
                self.metadata.use_mla, self.metadata.world_size
            )
            if (
                self.lmcache_worker is not None
                or self.use_layerwise
                or not self.save_only_first_rank
                or self.metadata.is_first_rank()
                or len(lookup_server_worker_ids) == 0
                or self.metadata.worker_id in lookup_server_worker_ids
            ):
                logger.info(
                    f"Initialize storage manager on rank {self.metadata.worker_id}, "
                    f"use layerwise: {self.use_layerwise},"
                    f"save only first rank: {self.save_only_first_rank}"
                )
                async_lookup_server = kwargs.get("async_lookup_server", None)
                self.storage_manager = StorageManager(
                    self.config,
                    self.metadata,
                    event_manager=self.event_manager,
                    lmcache_worker=self.lmcache_worker,
                    async_lookup_server=async_lookup_server,
                )
                if (getattr(self, "grouped_layers", False) or self.native_bulk_spec) and set(self.storage_manager.storage_backends) != {"LocalCPUBackend"}:
                    raise ValueError("Grouped/native SPEC CPU KV supports only a local CPU backend")
                # SpecAgent's scheduler runs in this same EngineCore process.
                # Expose the live CPU-cache probe only after this engine owns a
                # real storage manager.  Registering from __init__ is too early
                # and can be overwritten by scheduler-side LMCache objects that
                # do not have storage.
                builtins._specagent_lmcache_live_saved_tokens = (
                    self.specagent_live_saved_tokens
                )
                logger.info(
                    "[SpecAgent][CPU-LIVE-PROBE-LMCACHE] registered "
                    "engine_id=%s worker_id=%s retrieve_locations=%s "
                    "store_location=%s use_layerwise=%s",
                    id(self),
                    getattr(self.metadata, "worker_id", None),
                    self.retrieve_locations,
                    self.store_location,
                    self.use_layerwise,
                )
            self.post_inited = True

    def freeze(self, enabled: bool) -> None:
        """
        Set the freeze mode for the cache engine.

        When freeze mode is enabled:
        - All store operations will be skipped (no new data stored)
        - Only local_cpu backend will be used for retrieval
        - No admit/evict messages will be generated
        This protects the local_cpu hot cache from changes.

        Args:
            enabled (bool): Whether to enable freeze mode
        """
        if self.storage_manager is not None:
            self.storage_manager.set_freeze(enabled)

    def is_frozen(self) -> bool:
        """
        Get the current freeze mode status.

        Returns:
            bool: True if freeze mode is enabled, False otherwise
        """
        if self.storage_manager is not None:
            return self.storage_manager.is_frozen()
        return False

    def set_hot_cache(self, enabled: bool) -> None:
        """
        Dynamically enable or disable the LocalCPUBackend hot cache.

        When disabled, the existing hot cache entries will be cleared
        and no new data will be written to the hot cache.

        Args:
            enabled (bool): Whether to enable hot cache
        """
        if self.storage_manager is not None:
            self.storage_manager.set_hot_cache(enabled)

    def is_hot_cache_enabled(self) -> bool:
        """
        Get the current hot cache status of LocalCPUBackend.

        Returns:
            bool: True if hot cache is enabled, False otherwise
        """
        if self.storage_manager is not None:
            return self.storage_manager.is_hot_cache_enabled()
        return False

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def store(
        self,
        tokens: Optional[Union[torch.Tensor, list[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Store the tokens/hashes and mask into the cache engine.

        :param Optional[torch.Tensor] tokens: The tokens of the corresponding KV caches.

        :param Optional[List[int]] hashes: The hashes of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.
            Should include KV cache specific information (e.g., paged KV buffer
            and the page tables).

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.
        """
        # Health check: block operation if LMCache is unhealthy
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping store operation")
            return

        assert self.gpu_connector is not None, (
            "gpu_connector is required for store operation"
        )

        if self._is_passive():
            logger.debug(f"rank={self.metadata.worker_id} ignore store")
            return

        assert self.storage_manager is not None

        # Get req_id for logging
        req_id = self._get_req_id(kwargs)

        # Initialize num_to_store_tokens to avoid reference before assignment
        num_to_store_tokens = 0

        if mask is not None:
            num_to_store_tokens = torch.sum(mask).item()
        elif tokens is not None:
            num_to_store_tokens = len(tokens)
        elif hashes is not None:
            assert offsets is not None, (
                "Offsets should be set when hashes are provided during store"
            )
            num_to_store_tokens = sum(offsets)
            kwargs["slot_mapping"] = torch.tensor(
                kwargs["slot_mapping"], dtype=torch.long, device="cuda"
            )

        assert tokens is not None or hashes is not None, (
            "Either 'tokens' or 'hashes' must be provided."
        )

        # KVCache Check logging
        self._log_kvcache_for_check(
            operation="Store",
            kwargs=kwargs,
            token_count=num_to_store_tokens,
            require_req_id=False,
        )

        # Check if freeze mode is enabled
        if self.is_frozen():
            logger.debug(
                "Freeze mode enabled, skipping store operation for %d tokens",
                num_to_store_tokens,
            )
            return

        store_stats = self.stats_monitor.on_store_request(num_to_store_tokens)

        starts: List[int] = []
        ends: List[int] = []
        keys: List[CacheEngineKey] = []
        memory_objs: List[MemoryObj] = []

        tot_kv_size = 0
        tot_token_num = 0

        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        with store_stats.profile_process_tokens():
            prev_key = 0
            for start, end, key in self.token_database.process_tokens(
                tokens,
                hashes,
                offsets,
                mask,
                request_configs=request_configs,
            ):
                assert isinstance(key, CacheEngineKey)
                # Allocate the memory object
                num_tokens = end - start
                kv_shapes = self.metadata.get_shapes(num_tokens)
                kv_dtypes = self.metadata.get_dtypes()

                # TODO (Jiayi): should be batched in the future
                memory_obj = self.storage_manager.allocate(
                    kv_shapes,
                    kv_dtypes,
                    busy_loop=self.config.get_extra_config_value(
                        "force_store_wait", False
                    ),
                    fmt=self.fmt,
                )
                if memory_obj is None:
                    logger.warning(
                        "Local cpu memory under pressure so"
                        " choosing to store only "
                        f" {len(memory_objs)}"
                        " total chunks of KV cache."
                    )
                    break

                starts.append(start)
                ends.append(end)
                keys.append(key)
                memory_objs.append(memory_obj)
                tot_kv_size += memory_obj.get_size()
                tot_token_num += num_tokens

                # Create KV event
                if self.kv_events_enabled:
                    stored_event = CacheStoreEvent(
                        block_hashes=[key.chunk_hash],
                        parent_block_hash=None if start == 0 else prev_key,
                        token_ids=[],
                        block_size=num_tokens,
                        lora_id=None,
                        medium="cpu",
                        lora_name=None,
                    )
                    if tokens is not None:
                        stored_event.token_ids = convert_tokens_to_list(
                            tokens,
                            start,
                            end,
                        )
                        if isinstance(tokens, torch.Tensor):
                            stored_event.medium = tokens.device
                    elif hashes is not None:
                        stored_event.token_ids = hashes[start : end + 1]
                    logger.debug(
                        (
                            "Added kv cache event '%s' to kv cache events queue"
                            % stored_event
                        )
                    )
                    self.kv_events.append(stored_event)
                    prev_key = key.chunk_hash

        # memory_objs might be empty, directly return to avoid sending tokens
        if not memory_objs:
            return

        prepared_proofs = (
            _specagent_prepare_cpu_store_proofs(tokens, ends)
            if getattr(self, "native_bulk_spec", False) and tokens is not None
            else None
        )

        merge_source = None
        merge_prefix = int(kwargs.get("specagent_tail_merge_prefix_tokens", 0) or 0)
        chunk_size = int(getattr(self.token_database, "chunk_size", 0) or 0)
        merge_start = merge_prefix // chunk_size * chunk_size if chunk_size else -1
        if (getattr(self, "native_bulk_spec", False) and merge_prefix > merge_start
                and merge_start in starts):
            source_plan = list(self.token_database.process_tokens(
                tokens=tokens[:merge_prefix], request_configs=request_configs,
            ))
            if not source_plan or source_plan[-1][:2] != (merge_start, merge_prefix):
                raise RuntimeError("Native bulk partial prefix source is not covered")
            source_objects = self.storage_manager.batched_get(
                [source_plan[-1][2]], "LocalCPUBackend",
            )
            merge_source = source_objects[0]
            if merge_source is None:
                raise RuntimeError("Native bulk partial CPU prefix was evicted")
        try:
            with store_stats.profile_from_gpu():
                self.gpu_connector.batched_from_gpu(memory_objs, starts, ends, **kwargs)
            if merge_source is not None:
                target = memory_objs[starts.index(merge_start)]
                merge_native_bulk_prefix(merge_source, target, merge_prefix - merge_start)
        finally:
            if merge_source is not None:
                merge_source.ref_count_down()

        with store_stats.profile_put():
            transfer_spec = kwargs.get("transfer_spec", None)
            # TODO: we implicitly rely on batched_put to call ref_count_down
            # this management should be done in a cleaner way
            self.storage_manager.batched_put(
                keys,
                memory_objs,
                transfer_spec=transfer_spec,
                location=self.store_location,
            )

        if getattr(self, "native_bulk_spec", False) or os.environ.get("SPECAGENT_BULK_CPU_PROOF") == "1":
            if tokens is None:
                raise ValueError("SpecAgent CPU proof requires exact token IDs")
            _specagent_publish_cpu_store_proofs(
                tokens, starts, ends, int(kwargs.get("specagent_block_size", 16)),
                prepared=prepared_proofs,
            )

        self.stats_monitor.on_store_finished(
            store_stats,
            tot_token_num,
        )
        tot_time = store_stats.time_to_store()

        logger.info(
            "[req_id=%s] Stored %d out of total %d tokens. "
            "size: %.4f GB, cost %.4f ms, throughput: %.4f GB/s; "
            "offload_time: %.4f ms, put_time: %.4f ms",
            req_id,
            tot_token_num,
            num_to_store_tokens,
            tot_kv_size / 1024**3,
            tot_time * 1000,
            tot_kv_size / tot_time / 1024**3 if tot_time > 0 else 0,
            (store_stats.process_tokens_time + store_stats.from_gpu_time) * 1000,
            store_stats.put_time * 1000,
        )

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def store_layer(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Generator[None, None, None]:
        """
        Store the KV cache in a layerwise manner.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.

        return: A generator that yields None. In the first iteration, the
            generator allocates the memory objects for all layers and moves
            the KV cache of the first layer from GPU to CPU. In the next
            iterations, it moves the KV cache of layer i from GPU to the memory
            objects (on CPU) and puts the memory objects of layer i-1 to the
            storage backends. In the last iteration, it puts the memory objects
            of the last layer to the storage backends.
        """
        # Health check: block operation if LMCache is unhealthy
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping store_layer operation")
            return

        assert self.storage_manager is not None
        assert self.gpu_connector is not None, (
            "gpu_connector is required for store_layer operation"
        )

        # Get req_id for logging
        req_id = self._get_req_id(kwargs)

        if mask is not None:
            num_to_store_tokens = torch.sum(mask).item()
        else:
            num_to_store_tokens = len(tokens)

        # KVCache Check logging
        self._log_kvcache_for_check(
            operation="Layerwise store",
            kwargs=kwargs,
            token_count=num_to_store_tokens,
            require_req_id=True,
        )

        monitor_req_id = self.stats_monitor.on_store_request(num_to_store_tokens)

        # Check if freeze mode is enabled
        if self.is_frozen():
            logger.debug(
                "Freeze mode enabled, skipping store_layer for %d tokens",
                num_to_store_tokens,
            )
            # Still need to yield to avoid StopIteration
            for layer_id in range(self.num_layers):
                yield
            return

        grouped_layers = getattr(self, "grouped_layers", False)
        if grouped_layers and (
            kwargs.get("specagent_save_current_layer_only", False)
            or not kwargs.get("specagent_defer_layerwise_store_commit", False)
        ):
            raise ValueError("Grouped CPU KV requires all-layer completion before publication")
        group_keys = []
        starts = []
        ends = []
        keys = []
        memory_objs = []
        tot_token_num = 0
        kv_dtype = self.metadata.kv_dtype
        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        prev_key = 0
        merge_prefix_tokens = int(
            kwargs.get("specagent_tail_merge_prefix_tokens", 0) or 0
        )
        chunk_size = int(getattr(self.token_database, "chunk_size", 0) or 0)
        merge_chunk_start = (
            merge_prefix_tokens // chunk_size * chunk_size
            if chunk_size > 0 and merge_prefix_tokens % chunk_size
            else -1
        )
        merge_source_objs: Optional[List[MemoryObj]] = None
        if merge_chunk_start >= 0:
            source_key = None
            source_start = source_end = -1
            for source_start, source_end, candidate_key in (
                self.token_database.process_tokens(
                    tokens=tokens[:merge_prefix_tokens],
                    request_configs=request_configs,
                )
            ):
                source_key = candidate_key
            if (
                source_key is None
                or source_start != merge_chunk_start
                or source_end != merge_prefix_tokens
            ):
                raise RuntimeError(
                    "SpecAgent observation-tail merge could not identify the "
                    f"previous partial chunk: req={req_id} "
                    f"merge_prefix={merge_prefix_tokens}"
                )
            assert isinstance(source_key, CacheEngineKey)
            merge_source_objs = self.storage_manager.batched_get(
                source_key.split_layers(self.num_layers),
                self.retrieve_locations,
            )
            if any(obj is None for obj in merge_source_objs):
                for obj in merge_source_objs:
                    if obj is not None:
                        obj.ref_count_down()
                raise RuntimeError(
                    "SpecAgent observation-tail merge is missing the previous "
                    f"CPU partial chunk: req={req_id} "
                    f"chunk=[{merge_chunk_start},{merge_prefix_tokens})"
                )
            if os.environ.get("SPECAGENT_TAIL_SAVE_DEBUG"):
                logger.error(
                    "[SpecAgent][OBS-TAIL-MERGE] req=%s "
                    "chunk=[%d,%d) new_end=%d layers=%d",
                    req_id,
                    merge_chunk_start,
                    merge_prefix_tokens,
                    len(tokens),
                    len(merge_source_objs),
                )
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens, mask=mask, request_configs=request_configs
        ):
            assert isinstance(key, CacheEngineKey)

            keys_multi_layer = key.split_layers(self.num_layers)
            # Layerwise retrieval is only valid once every layer for this
            # chunk is present.  Checking layer 0 alone can advertise a chunk
            # while later layers are still being stored, which makes SpecAgent
            # scratch attention consume zero/stale KV for those layers.
            if all(
                self.storage_manager.contains(key_layer, self.retrieve_locations)
                for key_layer in ([key] if grouped_layers else keys_multi_layer)
            ):
                continue

            # Allocate the memory object
            num_tokens = end - start
            kv_shape_single_layer = self.gpu_connector.get_shape(num_tokens)

            memory_objs_multi_layer = self.storage_manager.batched_allocate(
                kv_shape_single_layer,
                kv_dtype,
                batch_size=self.num_layers,
                fmt=self.fmt,
                busy_loop=self.config.get_extra_config_value("force_store_wait", False),
            )

            if memory_objs_multi_layer is None:
                logger.warning(
                    "Local cpu memory under pressure so"
                    " choosing to not store the KV cache."
                )
                break

            group_keys.append(key)
            starts.append(start)
            ends.append(end)
            keys.append(keys_multi_layer)
            memory_objs.append(memory_objs_multi_layer)
            tot_token_num += num_tokens

            _spd = os.environ.get("LMCACHE_STOREPUT_DEBUG")
            if _spd:
                try:
                    import builtins as _b
                    _fh = getattr(_b, "_lmc_storeput_fh", None)
                    if _fh is None:
                        _fh = _b._lmc_storeput_fh = open(_spd, "a", buffering=1)
                        _b._lmc_storeput_n = 0
                    if _b._lmc_storeput_n < 12000:
                        _b._lmc_storeput_n += 1
                        _tk = tokens[start:end]
                        if hasattr(_tk, "tolist"):
                            _tk = _tk.tolist()
                        _fp = "n%d_f%d_l%d_s%d" % (
                            len(_tk), _tk[0] if _tk else 0,
                            _tk[-1] if _tk else 0, sum(_tk) & 0xffffff)
                        _fh.write(
                            "PUT ts=%.4f req=%s span=[%d:%d] tokfp=%s key0=%s\n" % (
                                time.monotonic(), req_id, start, end, _fp,
                                keys_multi_layer[0].to_string()[:80]))
                except Exception:
                    pass

            # Create KV event
            if self.kv_events_enabled and tokens is not None:
                stored_event = CacheStoreEvent(
                    block_hashes=[key.chunk_hash],
                    parent_block_hash=None if start == 0 else prev_key,
                    token_ids=[],
                    block_size=num_tokens,
                    lora_id=None,
                    medium="cpu",
                    lora_name=None,
                )
                if tokens is not None:
                    stored_event.token_ids = convert_tokens_to_list(
                        tokens,
                        start,
                        end,
                    )
                    if isinstance(tokens, torch.Tensor):
                        stored_event.medium = tokens.device
                logger.debug(
                    f"Added kv cache event '{stored_event}' to kv cache events queue"
                )
                self.kv_events.append(stored_event)
                prev_key = key.chunk_hash

        if keys:
            # Transpose the keys and memory objects into layer major format
            memory_objs = [list(row) for row in zip(*memory_objs, strict=False)]
            keys = [list(row) for row in zip(*keys, strict=False)]

            # Calculate total KV size for logging
            tot_kv_size = sum(
                mo.get_size() for layer_objs in memory_objs for mo in layer_objs
            )

            assert_layerwise_gpu_connector(self.gpu_connector)

            t_start = time.perf_counter()
            mem_obj_generator = self.gpu_connector.batched_from_gpu(
                memory_objs, starts, ends, **kwargs
            )
            store_current_layer_only = bool(
                kwargs.get("specagent_save_current_layer_only", False)
            )
            defer_layerwise_commit = bool(
                kwargs.get("specagent_defer_layerwise_store_commit", False)
            )
            if not store_current_layer_only:
                # The layerwise D2H generator yields immediately after
                # starting a layer transfer. The following next() waits for
                # that layer to finish before scheduling the next one. Prime it
                # once so each batched_put below sees completed CPU memory.
                next(mem_obj_generator)

            specagent_block_size = int(kwargs.get("specagent_block_size", 16))

            def commit_layer(layer_id: int) -> None:
                """Publish one fully copied CPU layer to the storage manager."""
                if merge_source_objs is not None:
                    source_obj = merge_source_objs[layer_id]
                    assert source_obj is not None
                    merged = False
                    for start, end, memory_obj in zip(
                        starts, ends, memory_objs[layer_id], strict=False
                    ):
                        if start != merge_chunk_start:
                            continue
                        prefix_len = merge_prefix_tokens - start
                        source_tensor = source_obj.tensor
                        target_tensor = memory_obj.tensor
                        assert source_tensor is not None
                        assert target_tensor is not None
                        if source_tensor.shape[0] != prefix_len:
                            raise RuntimeError(
                                "SpecAgent observation-tail source has wrong "
                                f"length: req={req_id} layer={layer_id} "
                                f"expected={prefix_len} "
                                f"actual={source_tensor.shape[0]}"
                            )
                        target_tensor[:prefix_len].copy_(source_tensor)
                        merged = True
                        break
                    source_obj.ref_count_down()
                    if not merged:
                        raise RuntimeError(
                            "SpecAgent observation-tail target chunk was not "
                            f"scheduled: req={req_id} layer={layer_id} "
                            f"chunk_start={merge_chunk_start}"
                        )
                with _lmc_record(
                    "lmcache store_layer: batched_put "
                    f"req={req_id} layer={layer_id} "
                    f"chunks={len(keys[layer_id])}"
                ):
                    if not grouped_layers:
                        self.storage_manager.batched_put(
                            keys[layer_id], memory_objs[layer_id], location=self.store_location,
                        )
                with _lmc_record(
                    "lmcache store_layer: fingerprint "
                    f"req={req_id} layer={layer_id} "
                    f"chunks={len(keys[layer_id])}"
                ):
                    for start, end, key, memory_obj in zip(
                        starts,
                        ends,
                        keys[layer_id],
                        memory_objs[layer_id],
                        strict=False,
                    ):
                        _specagent_record_actual_kv_fingerprint(
                            tokens,
                            memory_obj,
                            start,
                            end,
                            specagent_block_size,
                            req_id,
                            layer_id,
                            key,
                        )
                if layer_id == self.num_layers - 1:
                    if grouped_layers:
                        # Every layer has completed D2H and any sparse tail merge.
                        # One owner consumes the initial reference of each child;
                        # layer readers acquire independent references on demand.
                        groups = [LayerGroup(list(objects)) for objects in zip(*memory_objs, strict=True)]
                        self.storage_manager.batched_put(
                            group_keys, groups, location=self.store_location,
                        )
                    with _lmc_record(
                        "lmcache store_layer: cpu_store_proof "
                        f"req={req_id} layer={layer_id} "
                        f"chunks={len(keys[layer_id])}"
                    ):
                        # All chunks share the same immutable token prefix.
                        # Convert it once after the final layer completes; proof
                        # keys and their publication point remain unchanged.
                        _specagent_publish_cpu_store_proofs(
                            tokens, starts, ends, specagent_block_size
                        )

            for layer_id in range(self.num_layers):
                with _lmc_record(
                    "lmcache store_layer: d2h_wait "
                    f"req={req_id} layer={layer_id} "
                    f"chunks={len(keys[layer_id])}"
                ):
                    if store_current_layer_only:
                        # SpecAgent calls this generator after each model layer.
                        # Start and finish only the current layer here; do not
                        # advance into the next layer before its KV has been
                        # computed.
                        next(mem_obj_generator)
                        next(mem_obj_generator)
                    else:
                        next(mem_obj_generator)
                if not defer_layerwise_commit:
                    commit_layer(layer_id)

                yield

            if (
                store_current_layer_only
                and (
                    defer_layerwise_commit
                    or not os.environ.get("SPECAGENT_DISABLE_STORE_CLEANUP")
                )
            ):
                # Finish the GPU connector generator so its temporary transfer
                # buffer reaches ref_count_down(). In layerwise mode the
                # per-layer loop above stops immediately after the last layer
                # yield, leaving connector cleanup pending unless we drain it.
                next(mem_obj_generator, None)
            if defer_layerwise_commit:
                # batched_from_gpu synchronized its store stream during the
                # cleanup step above.  Only now expose CPU KV to retrievers.
                for layer_id in range(self.num_layers):
                    commit_layer(layer_id)

            tot_time = time.perf_counter() - t_start
            logger.info(
                "[req_id=%s] Stored %d out of total %d tokens. "
                "size: %.4f GB, cost %.4f ms, throughput: %.4f GB/s",
                req_id,
                tot_token_num,
                len(tokens),
                tot_kv_size / 1024**3,
                tot_time * 1000,
                tot_kv_size / tot_time / 1024**3 if tot_time > 0 else 0,
            )
        else:
            # If no cache are found, we still need to yield to avoid
            # `StopIteration`
            for layer_id in range(self.num_layers):
                yield

        self.stats_monitor.on_store_finished(monitor_req_id, tot_token_num)
        yield

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def retrieve(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Retrieve the KV caches from the cache engine. And put the retrieved
        KV cache to the serving engine via the GPU connector.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched,
            and the Falses will ALWAYS be at the PREFIX of the tensor.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.
            Should include KV cache specific information (e.g., paged KV buffer
            and the page tables).

        :return: the boolean mask indicating which tokens are retrieved. The
            length of the mask should be the same as the tokens. On CPU.

        :raises: ValueError if the number of Falses in the mask is not a
            multiple of the chunk size.
        """
        # Health check: block operation if LMCache is unhealthy
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping retrieve operation")
            return torch.zeros(len(tokens), dtype=torch.bool)

        assert self.gpu_connector is not None, (
            "gpu_connector is required for retrieve operation"
        )

        # Get req_id for logging
        req_id = self._get_req_id(kwargs)

        tot_kv_size = 0

        if mask is not None:
            num_required_tokens = torch.sum(mask).item()
        else:
            num_required_tokens = len(tokens)

        # KVCache Check logging
        self._log_kvcache_for_check(
            operation="retrieve",
            kwargs=kwargs,
            token_count=num_required_tokens,
            require_req_id=True,
        )

        retrieve_stats = self.stats_monitor.on_retrieve_request(num_required_tokens)

        ret_mask = torch.zeros(len(tokens), dtype=torch.bool, device="cpu")

        reordered_chunks: List[ProcessedChunk] = []
        if not self._is_passive():
            with retrieve_stats.profile_process_tokens():
                if self.async_loading:
                    reordered_chunks, tot_kv_size = self._async_process_tokens_internal(  # noqa: E501
                        tokens,
                        mask,
                        ret_mask,
                        **kwargs,
                    )
                else:
                    reordered_chunks, tot_kv_size = self._process_tokens_internal(
                        tokens,
                        mask,
                        ret_mask,
                        **kwargs,
                    )

        if self.save_only_first_rank:
            with retrieve_stats.profile_broadcast():
                with torch.cuda.stream(self.broadcast_stream):
                    self._broadcast_or_receive_memory_objs(
                        reordered_chunks,
                        ret_mask,
                    )

                # if self.gpu_connector has load_stream, self.broadcast_stream is equals
                # to self.gpu_connector.load_stream, the broadcast and to_gpu operation
                # will execute sequentially within the stream.
                # if self.gpu_connector does not have load_stream, self.broadcast_stream
                # is created by torch.cuda.Stream(), we need to synchronize broadcast
                # operation, and then process to_cpu operation.
                if not hasattr(self.gpu_connector, "load_stream"):
                    self.broadcast_stream.synchronize()

        # NOTE(Jiayi): memory_obj doesn't have to be a pinned
        # cpu tensor for the sake of performance.
        # For example, disk->gpu is faster than disk->cpu->gpu.
        # RDMA is another example.
        if len(reordered_chunks) > 0:
            with retrieve_stats.profile_to_gpu():
                _, memory_objs, starts, ends = zip(*reordered_chunks, strict=False)
                self.gpu_connector.batched_to_gpu(
                    list(memory_objs), list(starts), list(ends), **kwargs
                )

        # TODO(Jiayi): Remove the following for loop with batched operations
        # TODO(Jiayi): Need to refactor the `remove_after_retrieve` logic.
        for key, memory_obj, _, _ in reordered_chunks:
            if self.remove_after_retrieve and not self._is_passive():
                assert self.storage_manager is not None
                self.storage_manager.remove(key, self.retrieve_locations)
            if not self.async_loading:
                memory_obj.ref_count_down()

        retrieved_tokens = torch.sum(ret_mask)
        self.stats_monitor.on_retrieve_finished(
            retrieve_stats,
            retrieved_tokens,
        )
        onload_time = retrieve_stats.time_to_retrieve()
        # The retrieved may be larger than the need_to_load
        # Example (page_size=16, chunk_size=256):
        #
        # chunks:  [0..255]                [256..511]
        # pages:   [0..15]...[240..255]    [256..271][272..287] ...
        #
        # num_computed_tokens = 288 => vLLM already has [0..287] (18 pages)
        # LMCache hit_prefix_tokens = 512 => cache covers [0..511] (2 chunks)
        #
        # Skip chunk 1, retrieve chunk 2, overwrite [256..287] (32-token overlap)
        # need_to_load: 512 - 288 = 224 tokens
        # retrieved: 256 tokens
        if not self._is_passive():
            logger.info(
                "[req_id=%s] Retrieved %d out of %d required tokens "
                "(from %d total tokens). size: %.4f gb, "
                "cost %.4f ms, throughput: %.4f GB/s;",
                req_id,
                retrieved_tokens,
                num_required_tokens,
                len(tokens),
                tot_kv_size / 1024**3,
                onload_time * 1000,
                tot_kv_size / onload_time / 1024**3 if onload_time > 0 else 0,
            )
        return ret_mask

    def _specagent_memo_chunk_keys(
        self,
        tokens: Union[torch.Tensor, list[int]],
        max_end: int,
        request_configs: Optional[dict],
        memo_id: str,
    ) -> list[tuple[int, int, CacheEngineKey]]:
        """process_tokens(tokens[:max_end]) with a per-request hash-chain memo.

        Memo entry: (kind, token prefix, full-chunk hashes, keys, configs).
        Only full chunks are memoized; the reused prefix is the longest
        chunk-aligned prefix identical to the previous call of this request.
        """
        import numpy as np
        from collections import OrderedDict

        db = self.token_database
        chunk = int(db.chunk_size)
        memo = getattr(self, "_specagent_prefix_hash_memo", None)
        if memo is None:
            memo = self._specagent_prefix_hash_memo = OrderedDict()
        cached = memo.get(memo_id)
        if isinstance(tokens, torch.Tensor):
            kind = "np"
            seq = tokens[:max_end].detach().cpu().numpy().astype(np.int32, copy=False)
        else:
            kind = "list"
            seq = list(tokens[:max_end])
        total = len(seq)
        reuse = 0
        hashes: list[int] = []
        keys: list[CacheEngineKey] = []
        if cached is not None and cached[0] == kind:
            _, old, old_hashes, old_keys, old_configs = cached
            n = min(len(old), total, len(old_hashes) * chunk)
            n -= n % chunk
            if kind == "np":
                diff = np.flatnonzero(old[:n] != seq[:n])
                same = int(diff[0]) if diff.size else n
            elif old[:n] == seq[:n]:
                same = n
            else:
                same = 0
                while same < n and old[same:same + chunk] == seq[same:same + chunk]:
                    same += chunk
            reuse = same // chunk
            hashes = old_hashes[:reuse]
            if old_configs == request_configs:
                keys = old_keys[:reuse]
        prefix = hashes[-1] if hashes else db._get_init_hash()
        save_unfull = (
            db.config.save_unfull_chunk if db.config is not None else True
        )
        end = total if save_unfull else total - total % chunk
        out: list[tuple[int, int, CacheEngineKey]] = []
        full_hashes = list(hashes)
        full_keys = list(keys)
        for chunk_id in range(0, (end + chunk - 1) // chunk):
            start = chunk_id * chunk
            stop = min(start + chunk, total)
            if chunk_id < reuse:
                hash_val = hashes[chunk_id]
                key = keys[chunk_id] if chunk_id < len(keys) else None
            else:
                piece = seq[start:stop]
                hash_val = db._hash_tokens(
                    tuple(piece.tolist()) if kind == "np" else tuple(piece), prefix,
                )
                key = None
            if key is None:
                key = db._make_key_by_hash(hash_val, request_configs)
            if stop - start == chunk and chunk_id >= reuse:
                full_hashes.append(hash_val)
            if stop - start == chunk and chunk_id >= len(keys):
                full_keys.append(key)
            prefix = hash_val
            out.append((start, stop, key))
        memo[memo_id] = (kind, seq, full_hashes, full_keys[: len(full_hashes)], request_configs)
        memo.move_to_end(memo_id)
        while len(memo) > 96:
            memo.popitem(last=False)
        return out

    def plan_sparse_layerwise_chunks(
        self,
        tokens: Union[torch.Tensor, list[int]],
        segments: Iterable[tuple[int, int]],
        request_configs: Optional[dict] = None,
        memo_id: Optional[str] = None,
    ) -> list[list[tuple[int, int, CacheEngineKey]]]:
        """Build cache keys for multiple sparse retrieval segments once.

        Args:
            tokens: Complete token prefix containing every segment.
            segments: Sorted, non-overlapping ``[start, end)`` token ranges.
            request_configs: Optional request-specific cache-key inputs.
        Returns:
            One chunk plan per input segment. Each plan contains the same
            ``(start, end, key)`` tuples that an independent prefix-masked
            ``process_tokens`` call would produce.

        Raises:
            ValueError: If a segment is invalid, overlapping, or outside the
                supplied token prefix.

        Notes:
            Prefix hashes are chained, so independently planning sparse
            segments repeats all earlier hashing. This method performs one
            linear prefix scan and partitions the resulting chunk keys.
        """
        normalized_segments = [(int(start), int(end)) for start, end in segments]
        previous_end = 0
        for index, (start, end) in enumerate(normalized_segments):
            if start < 0 or start >= end or end > len(tokens):
                raise ValueError(
                    f"invalid sparse retrieval segment [{start}, {end})"
                )
            if index > 0 and start < previous_end:
                raise ValueError("sparse retrieval segments must not overlap")
            previous_end = end
        if not normalized_segments:
            return []

        # Per-request whole-plan reuse (SPECAGENT_VERIFY_CHUNKPLAN_CACHE=1):
        # the fetched segments lie in the committed prefix, whose tokens never
        # change for a request id, so identical (request, segments) yields an
        # identical plan. CHECK mode recomputes and compares every hit.
        plan_cache = None
        cache_key = None
        if memo_id is not None and os.environ.get(
            "SPECAGENT_VERIFY_CHUNKPLAN_CACHE"
        ) == "1":
            from collections import OrderedDict

            plan_cache = getattr(self, "_specagent_chunk_plan_cache", None)
            if plan_cache is None:
                plan_cache = self._specagent_chunk_plan_cache = OrderedDict()
            cache_key = (memo_id, tuple(normalized_segments), request_configs is None)
            cached_plans = plan_cache.get(cache_key)
            if cached_plans is not None and cached_plans[0] == request_configs:
                plan_cache.move_to_end(cache_key)
                if os.environ.get("SPECAGENT_VERIFY_PLAN_CACHE_CHECK") == "1":
                    os.environ["SPECAGENT_VERIFY_CHUNKPLAN_CACHE"] = "0"
                    try:
                        fresh = self.plan_sparse_layerwise_chunks(
                            tokens, normalized_segments, request_configs, memo_id
                        )
                    finally:
                        os.environ["SPECAGENT_VERIFY_CHUNKPLAN_CACHE"] = "1"
                    if fresh != cached_plans[1]:
                        raise AssertionError("[SpecAgent] chunk plan cache mismatch")
                return [list(p) for p in cached_plans[1]]

        plans: list[list[tuple[int, int, CacheEngineKey]]] = [
            [] for _ in normalized_segments
        ]
        segment_index = 0
        max_end = normalized_segments[-1][1]
        use_memo = (
            memo_id is not None
            and os.environ.get("SPECAGENT_PREFIX_HASH_MEMO", "1") == "1"
            and hasattr(self.token_database, "_prefix_hash")
            and type(self.token_database).__name__ == "ChunkedTokenDatabase"
        )
        if use_memo:
            chunk_results = self._specagent_memo_chunk_keys(
                tokens, max_end, request_configs, memo_id,
            )
            if os.environ.get("SPECAGENT_FASTPATH_VERIFY") == "1":
                reference = list(self.token_database.process_tokens(
                    tokens=tokens[:max_end], request_configs=request_configs,
                ))
                if reference != chunk_results:
                    logger.error("[SpecAgent][FASTPATH-MISMATCH] prefix hash memo")
                    raise AssertionError("fastpath prefix hash memo mismatch")
        else:
            chunk_results = self.token_database.process_tokens(
                tokens=tokens[:max_end],
                request_configs=request_configs,
            )
        for start, end, key in chunk_results:
            assert isinstance(key, CacheEngineKey)
            while (
                segment_index < len(normalized_segments)
                and start >= normalized_segments[segment_index][1]
            ):
                segment_index += 1
            if segment_index >= len(normalized_segments):
                break
            segment_start, segment_end = normalized_segments[segment_index]
            if start >= segment_start and end <= segment_end:
                plans[segment_index].append((start, end, key))
        if plan_cache is not None:
            plan_cache[cache_key] = (request_configs, [list(p) for p in plans])
            plan_cache.move_to_end(cache_key)
            while len(plan_cache) > 32:
                plan_cache.popitem(last=False)
        return plans

    def bulk_validate_sparse_layerwise_plan(
        self,
        plans: list[list[tuple[int, int, CacheEngineKey]]],
    ) -> Optional[
        tuple[list[int], list[int], list[list[CacheEngineKey]], str]
    ]:
        """Validate a complete sparse layerwise plan with one storage lookup.

        Args:
            plans: Segment-partitioned base cache keys from
                :meth:`plan_sparse_layerwise_chunks`.

        Returns:
            Chunk starts, ends, per-chunk layer keys, and the single backing
            location when every layer key is present in one backend. Returns
            ``None`` for partial or multi-backend plans so callers can use the
            legacy chunk-by-chunk fallback.
        """
        starts: list[int] = []
        ends: list[int] = []
        keys_by_chunk: list[list[CacheEngineKey]] = []
        flat_keys: list[CacheEngineKey] = []
        native = bool(getattr(self, "native_bulk_spec", False))
        for plan in plans:
            for start, end, key in plan:
                layer_keys = (
                    _LazyLayerKeys(key, self.num_layers) if native
                    else key.split_layers(self.num_layers)
                )
                starts.append(int(start))
                ends.append(int(end))
                keys_by_chunk.append(layer_keys)
                # A grouped owner is published only after every layer is
                # complete; repeating its membership check per layer adds no
                # validity evidence.
                if getattr(self, "grouped_layers", False) or getattr(self, "native_bulk_spec", False):
                    flat_keys.append(key)
                else:
                    flat_keys.extend(layer_keys)
        if not flat_keys:
            return None
        hit_layers, block_mapping = self.storage_manager.batched_contains(
            flat_keys,
            getattr(self, "retrieve_locations", None),
        )
        if hit_layers != len(flat_keys) or len(block_mapping) != 1:
            return None
        return (
            starts,
            ends,
            keys_by_chunk,
            next(iter(block_mapping.keys())),
        )

    def prepare_sparse_layerwise_plans(
        self,
        plans: list[list[tuple[int, int, CacheEngineKey]]],
    ) -> Optional[SpecAgentPreparedLayerwiseBatch]:
        """Resolve a sparse plan entirely on CPU for later GPU consumption.

        The worker performs the single membership query and resolves all
        per-layer MemoryObjs.  It deliberately does not create CUDA events,
        touch streams, or enqueue copies, so it is safe to run in a regular
        Python executor while the model thread finishes request setup.
        """
        validated = self.bulk_validate_sparse_layerwise_plan(plans)
        if validated is None:
            return None
        starts, ends, keys_by_chunk, location = validated
        backend = self.storage_manager.storage_backends.get(location)
        if backend is None or not hasattr(backend, "batched_get_blocking"):
            return None

        source_owners: Optional[list[MemoryObj]] = None
        if getattr(self, "native_bulk_spec", False):
            owner_keys = [
                chunk_key(keys.layer0() if isinstance(keys, _LazyLayerKeys) else keys[0])
                for keys in keys_by_chunk
            ]
            if os.environ.get("SPECAGENT_FASTPATH_VERIFY") == "1":
                reference_keys = [
                    chunk_key(
                        keys.key.split_layers(self.num_layers)[0]
                        if isinstance(keys, _LazyLayerKeys) else keys[0]
                    )
                    for keys in keys_by_chunk
                ]
                if reference_keys != owner_keys:
                    logger.error("[SpecAgent][FASTPATH-MISMATCH] native owner keys")
                    raise AssertionError("fastpath native owner keys mismatch")
            owners = backend.batched_get_blocking(owner_keys)
            acquired = [owner for owner in owners if owner is not None]
            try:
                if len(acquired) != len(owners):
                    raise RuntimeError("Native bulk CPU owner evicted during acquisition")
                resolved_layers = borrow_bulk_layer_views(acquired, self.num_layers)
            except BaseException:
                for owner in acquired:
                    owner.ref_count_down()
                raise
            # Transfer chunk leases through the plan until GPU completion.
            source_owners = acquired
        elif getattr(self, "grouped_layers", False):
            resolved_layers = backend.batched_get_layer_groups(
                [keys[0] for keys in keys_by_chunk], self.num_layers,
            )
            if resolved_layers is None:
                raise RuntimeError(
                    "[SpecAgent][ASYNC-PREPARE] CPU layer group disappeared "
                    "between validation and acquisition"
                )
        else:
            keys_layer_major = [
                list(row) for row in zip(*keys_by_chunk, strict=False)
            ]
            resolved_layers: list[list[MemoryObj]] = []
            try:
                for layer_id, keys_layer in enumerate(keys_layer_major):
                    raw_layer = backend.batched_get_blocking(keys_layer)
                    acquired = [obj for obj in raw_layer if obj is not None]
                    if len(acquired) != len(keys_layer):
                        # Include successful gets from this partial layer in the
                        # rollback; each one already owns a reference.
                        for mem_obj in acquired:
                            mem_obj.ref_count_down()
                        raise RuntimeError(
                            "[SpecAgent][ASYNC-PREPARE] CPU MemoryObj missing "
                            f"at layer={layer_id}: evicted between contains and "
                            "resolve"
                        )
                    resolved_layers.append(acquired)
            except Exception:
                for prior_layer in resolved_layers:
                    for mem_obj in prior_layer:
                        mem_obj.ref_count_down()
                raise
        prepared_segments: list[SpecAgentPreparedLayerwiseSegment] = []
        offset = 0
        for plan in plans:
            next_offset = offset + len(plan)
            prepared_segments.append(
                SpecAgentPreparedLayerwiseSegment(
                    starts=starts[offset:next_offset],
                    ends=ends[offset:next_offset],
                    keys_by_chunk=keys_by_chunk[offset:next_offset],
                    location=location,
                    memory_objs_by_layer=[
                        layer[offset:next_offset] for layer in resolved_layers
                    ],
                    source_owners=(
                        source_owners[offset:next_offset]
                        if source_owners is not None else None
                    ),
                )
            )
            offset = next_offset
        return SpecAgentPreparedLayerwiseBatch(prepared_segments)

    def _specagent_prepare_with_plan_cache(
        self,
        req_id: str,
        plan: list[tuple[int, int, CacheEngineKey]],
    ) -> Optional[SpecAgentPreparedLayerwiseBatch]:
        """prepare_sparse_layerwise_plans([plan]) with a per-request reuse cache.

        Within one SPEC episode every VERIFY round of a request fetches the
        same missing chunks, so the membership check, CPU owner acquisition
        and borrowed layer-view construction are identical. The cache keeps
        one extra reference on each native owner (so the CPU backend cannot
        evict or free it: can_evict requires ref_count == 1) and hands each
        reuse a fresh lease that the normal consumer releases at GPU
        completion. Only an exact (start, end, key) plan match is reused;
        any other plan for the request replaces the entry. Entries are
        bounded by request count and total tokens (LRU) and release their
        references when dropped.

        Env: SPECAGENT_VERIFY_PLAN_CACHE_MAX_REQS (8),
        SPECAGENT_VERIFY_PLAN_CACHE_MAX_TOKENS (393216),
        SPECAGENT_VERIFY_PLAN_CACHE_CHECK=1 compares every hit with a fresh
        preparation and raises on any difference.
        """
        from collections import OrderedDict

        cache = getattr(self, "_specagent_verify_plan_cache", None)
        if cache is None:
            cache = self._specagent_verify_plan_cache = OrderedDict()
        stats = vars(builtins).setdefault(
            "_specagent_verify_plan_cache_stats",
            {"hit": 0, "miss": 0, "replace": 0, "evict": 0, "uncacheable": 0},
        )
        sig = tuple((int(s), int(e), k) for s, e, k in plan)
        entry = cache.get(req_id)
        stats["calls"] = stats.get("calls", 0) + 1
        if stats["calls"] <= 30:
            if entry is None:
                why = "new_req"
            elif entry["sig"] != sig:
                old = entry["sig"]
                first = next((i for i, (a, b) in enumerate(zip(old, sig)) if a != b),
                             min(len(old), len(sig)))
                why = (f"sig_diff len {len(old)}->{len(sig)} first_diff={first} "
                       f"old={old[first][:2] if first < len(old) else None} "
                       f"new={sig[first][:2] if first < len(sig) else None}")
            else:
                why = "hit_candidate"
            logger.warning("[SpecAgent][VERIFY-PLAN-CACHE-TRACE] req=%s chunks=%d %s",
                           req_id, len(sig), why)
        if (entry is not None and entry["sig"] != sig and sig
                and os.environ.get("SPECAGENT_VERIFY_PLAN_CACHE_SUBSET", "1") == "1"
                and all(owner.is_valid() for owner in entry["owners"])):
            # A plan that keeps a subset of the cached chunks (same
            # (start, end, key) each, in order; e.g. after a reclaim claimed
            # some of its holes) reuses the cached owners and descriptors:
            # drop the leases of the chunks that left, keep the rest.
            pos = entry.get("pos")
            if pos is None:
                pos = entry["pos"] = {c: i for i, c in enumerate(entry["sig"])}
            idx = [pos.get(c) for c in sig]
            if None not in idx and all(a < b for a, b in zip(idx, idx[1:])):
                from lmcache.v1.specagent_bulk_view import subset_layer_columns

                keep = set(idx)
                for i, owner in enumerate(entry["owners"]):
                    if i not in keep:
                        owner.ref_count_down()
                entry["owners"] = [entry["owners"][i] for i in idx]
                entry["columns"] = subset_layer_columns(entry["columns"], idx)
                entry["starts"] = [entry["starts"][i] for i in idx]
                entry["ends"] = [entry["ends"][i] for i in idx]
                entry["keys_by_chunk"] = [entry["keys_by_chunk"][i] for i in idx]
                l0 = entry.get("layer0")
                entry["layer0"] = None if l0 is None else (
                    [l0[0][i] for i in idx], [l0[1][i] for i in idx])
                entry["tokens"] = sum(
                    e - s for s, e in zip(entry["starts"], entry["ends"]))
                entry["sig"] = sig
                entry["pos"] = None
                stats["hit_subset"] = stats.get("hit_subset", 0) + 1
        if entry is not None and entry["sig"] == sig and all(
            owner.is_valid() for owner in entry["owners"]
        ):
            for owner in entry["owners"]:
                owner.ref_count_up()
            segment = SpecAgentPreparedLayerwiseSegment(
                starts=entry["starts"],
                ends=entry["ends"],
                keys_by_chunk=entry["keys_by_chunk"],
                location=entry["location"],
                memory_objs_by_layer=entry["columns"],
                source_owners=list(entry["owners"]),
                layer0_sources=entry.get("layer0"),
            )
            cache.move_to_end(req_id)
            entry["last"] = stats["calls"]
            stats["hit"] += 1
            if stats["hit"] % 200 == 0:
                logger.warning("[SpecAgent][VERIFY-PLAN-CACHE] %s entries=%d", stats, len(cache))
            if os.environ.get("SPECAGENT_VERIFY_PLAN_CACHE_CHECK") == "1":
                fresh = self.prepare_sparse_layerwise_plans([plan])
                try:
                    if fresh is None:
                        raise AssertionError("plan cache hit but fresh plan is None")
                    f = fresh.claim(0)
                    same = (
                        f.starts == segment.starts
                        and f.ends == segment.ends
                        and f.location == segment.location
                        and len(f.source_owners or []) == len(segment.source_owners)
                        and all(a is b for a, b in zip(
                            f.source_owners or [], segment.source_owners))
                    )
                    f.release_source_owners()
                    if not same:
                        raise AssertionError("[SpecAgent] verify plan cache mismatch")
                finally:
                    if fresh is not None:
                        fresh.release_unclaimed()
            return SpecAgentPreparedLayerwiseBatch([segment])
        if entry is not None:
            self._specagent_plan_cache_drop(req_id)
            stats["replace"] += 1
        stats["miss"] += 1
        evicted_ids = vars(builtins).setdefault(
            "_specagent_verify_plan_cache_evicted_ids", set())
        big = len(sig) * (self.token_database.chunk_size
                          if getattr(self, "token_database", None) else 0) > 65536
        why = ("sig" if entry is not None and entry["sig"] != sig
               else "invalid" if entry is not None
               else "evicted" if req_id in evicted_ids else "new")
        evicted_ids.discard(req_id)
        rk = "why_" + why + ("_big" if big else "")
        stats[rk] = stats.get(rk, 0) + 1
        if (os.environ.get("SPECAGENT_VERIFY_PLAN_CACHE_STATS") == "1"
                and stats["miss"] % 25 == 0):
            logger.error(
                "[SpecAgent][VERIFY-PLAN-CACHE-STATS] %s entries=%d tokens=%d",
                stats, len(cache), sum(v["tokens"] for v in cache.values()))
        prepared = self.prepare_sparse_layerwise_plans([plan])
        if prepared is None:
            return None
        segment = prepared.claim(0)
        owners = segment.source_owners
        if owners is None:
            stats["uncacheable"] += 1
            return SpecAgentPreparedLayerwiseBatch([segment])
        max_reqs = int(os.environ.get("SPECAGENT_VERIFY_PLAN_CACHE_MAX_REQS", "8"))
        max_tokens = int(
            os.environ.get("SPECAGENT_VERIFY_PLAN_CACHE_MAX_TOKENS", "393216")
        )
        new_tokens = sum(e - s for s, e in zip(segment.starts, segment.ends))
        if os.environ.get("SPECAGENT_VERIFY_PLAN_CACHE_ADMIT") == "keep":
            # Keep-resident admission (same caps as LRU): cyclic access by more
            # large SPEC requests than fit makes LRU evict exactly the entry
            # needed next. Evict only entries unused for STALE_CALLS calls;
            # if the new plan still does not fit, serve it uncached.
            stale_calls = int(os.environ.get(
                "SPECAGENT_VERIFY_PLAN_CACHE_STALE_CALLS", "24"))
            now = stats["calls"]

            def _fits() -> bool:
                return not cache or (
                    len(cache) + 1 <= max_reqs
                    and sum(v["tokens"] for v in cache.values()) + new_tokens
                    <= max_tokens
                )

            for old_id in [k for k, v in cache.items()
                           if now - v.get("last", now) > stale_calls]:
                if _fits():
                    break
                self._specagent_plan_cache_drop(old_id)
                stats["evict_stale"] = stats.get("evict_stale", 0) + 1
                vars(builtins).setdefault(
                    "_specagent_verify_plan_cache_evicted_ids", set()).add(old_id)
            if not _fits():
                stats["not_admitted"] = stats.get("not_admitted", 0) + 1
                vars(builtins).setdefault(
                    "_specagent_verify_plan_cache_evicted_ids", set()).add(req_id)
                return SpecAgentPreparedLayerwiseBatch([segment])
        for owner in owners:
            owner.ref_count_up()
        layer0 = None
        if (os.environ.get("SPECAGENT_VERIFY_SCHED_FAST") == "1"
                and segment.memory_objs_by_layer):
            from lmcache.v1.specagent_bulk_view import layer0_sources

            layer0 = layer0_sources(
                segment.memory_objs_by_layer[0], self.num_layers)
            segment.layer0_sources = layer0
        cache[req_id] = dict(
            sig=sig, owners=list(owners), columns=segment.memory_objs_by_layer,
            starts=segment.starts, ends=segment.ends,
            keys_by_chunk=segment.keys_by_chunk, location=segment.location,
            tokens=new_tokens, last=stats["calls"], layer0=layer0,
        )
        while len(cache) > 1 and (
            len(cache) > max_reqs
            or sum(v["tokens"] for v in cache.values()) > max_tokens
        ):
            oldest = next(iter(cache))
            self._specagent_plan_cache_drop(oldest)
            stats["evict"] += 1
            vars(builtins).setdefault(
                "_specagent_verify_plan_cache_evicted_ids", set()).add(oldest)
        return SpecAgentPreparedLayerwiseBatch([segment])

    def _specagent_prepare_with_weak_plan_cache(
        self,
        req_id: str,
        plan: list[tuple[int, int, CacheEngineKey]],
    ) -> Optional[SpecAgentPreparedLayerwiseBatch]:
        """_specagent_prepare_with_plan_cache without pinning
        (SPECAGENT_VERIFY_SCHED_FAST=1).

        Entries keep no owner reference, so the CPU cache evicts as usual and
        no token cap is needed (the pinned cache admitted one long-context
        plan and served the rest uncached). A hit re-acquires the owners
        through the backend under its lock (batched_get_blocking, the same
        leases prepare_sparse_layerwise_plans takes) and requires the
        identical live objects for every chunk; an evicted or re-stored
        chunk is a miss. The cached borrowed layer columns and layer-0
        sources are descriptors of those objects. A plan that keeps a subset
        of the cached chunks (same (start, end, key) each, in order) hits on
        that subset. Entries: SPECAGENT_VERIFY_PLAN_CACHE_MAX_REQS (16), LRU.
        SPECAGENT_VERIFY_PLAN_CACHE_CHECK=1 compares every hit with a fresh
        preparation.
        """
        from collections import OrderedDict

        from lmcache.v1.specagent_bulk_view import (
            layer0_sources,
            subset_layer_columns,
        )

        cache = getattr(self, "_specagent_verify_weak_plan_cache", None)
        if cache is None:
            cache = self._specagent_verify_weak_plan_cache = OrderedDict()
        stats = vars(builtins).setdefault(
            "_specagent_verify_plan_cache_stats",
            {"hit": 0, "miss": 0, "replace": 0, "evict": 0, "uncacheable": 0},
        )
        stats["calls"] = stats.get("calls", 0) + 1
        stats["weak"] = 1
        sig = tuple((int(s), int(e), k) for s, e, k in plan)
        entry = cache.get(req_id)
        idx: Optional[list[int]] = None
        if entry is not None and sig:
            if entry["sig"] == sig:
                idx = list(range(len(sig)))
            else:
                pos = entry.get("pos")
                if pos is None:
                    pos = entry["pos"] = {c: i for i, c in enumerate(entry["sig"])}
                cand = [pos.get(c) for c in sig]
                if None not in cand and all(
                    a < b for a, b in zip(cand, cand[1:])
                ):
                    idx = cand  # type: ignore[assignment]
        if idx is not None:
            assert entry is not None
            backend = self.storage_manager.storage_backends.get(entry["location"])
            owners = None
            if backend is not None and hasattr(backend, "batched_get_blocking"):
                owners = backend.batched_get_blocking(
                    [entry["owner_keys"][i] for i in idx])
            cached_owners = entry["owners"]
            if owners is not None and all(
                o is not None and o is cached_owners[i] and o.is_valid()
                for o, i in zip(owners, idx, strict=True)
            ):
                if len(idx) != len(entry["sig"]):
                    entry["owners"] = [cached_owners[i] for i in idx]
                    entry["owner_keys"] = [entry["owner_keys"][i] for i in idx]
                    entry["columns"] = subset_layer_columns(entry["columns"], idx)
                    entry["starts"] = [entry["starts"][i] for i in idx]
                    entry["ends"] = [entry["ends"][i] for i in idx]
                    entry["keys_by_chunk"] = [
                        entry["keys_by_chunk"][i] for i in idx]
                    l0 = entry.get("layer0")
                    entry["layer0"] = None if l0 is None else (
                        [l0[0][i] for i in idx], [l0[1][i] for i in idx])
                    entry["sig"] = sig
                    entry["pos"] = None
                    stats["hit_subset"] = stats.get("hit_subset", 0) + 1
                segment = SpecAgentPreparedLayerwiseSegment(
                    starts=entry["starts"],
                    ends=entry["ends"],
                    keys_by_chunk=entry["keys_by_chunk"],
                    location=entry["location"],
                    memory_objs_by_layer=entry["columns"],
                    source_owners=list(owners),
                    layer0_sources=entry.get("layer0"),
                )
                cache.move_to_end(req_id)
                entry["last"] = stats["calls"]
                stats["hit"] += 1
                if os.environ.get("SPECAGENT_VERIFY_PLAN_CACHE_CHECK") == "1":
                    fresh = self.prepare_sparse_layerwise_plans([plan])
                    try:
                        if fresh is None:
                            raise AssertionError(
                                "weak plan cache hit but fresh plan is None")
                        f = fresh.claim(0)
                        same = (
                            f.starts == segment.starts
                            and f.ends == segment.ends
                            and f.location == segment.location
                            and len(f.source_owners or []) == len(segment.source_owners)
                            and all(a is b for a, b in zip(
                                f.source_owners or [], segment.source_owners))
                            and len(f.memory_objs_by_layer) == len(segment.memory_objs_by_layer)
                            and all(
                                len(fc) == len(sc) and all(
                                    fm is sm or (
                                        hasattr(fm, "tensor")
                                        and fm.tensor.data_ptr()
                                        == sm.tensor.data_ptr())
                                    for fm, sm in zip(fc, sc, strict=True))
                                for fc, sc in zip(
                                    f.memory_objs_by_layer[:2],
                                    segment.memory_objs_by_layer[:2], strict=True))
                        )
                        f.release_source_owners()
                        if not same:
                            raise AssertionError(
                                "[SpecAgent] weak verify plan cache mismatch")
                    finally:
                        if fresh is not None:
                            fresh.release_unclaimed()
                return SpecAgentPreparedLayerwiseBatch([segment])
            if owners:
                for o in owners:
                    if o is not None:
                        o.ref_count_down()
            stats["why_changed"] = stats.get("why_changed", 0) + 1
        elif entry is not None:
            stats["why_sig"] = stats.get("why_sig", 0) + 1
        else:
            stats["why_new"] = stats.get("why_new", 0) + 1
        if entry is not None:
            cache.pop(req_id, None)
            stats["replace"] += 1
        stats["miss"] += 1
        prepared = self.prepare_sparse_layerwise_plans([plan])
        if prepared is None:
            return None
        segment = prepared.claim(0)
        owners = segment.source_owners
        if owners is None:
            stats["uncacheable"] += 1
            return SpecAgentPreparedLayerwiseBatch([segment])
        layer0 = None
        if segment.memory_objs_by_layer:
            layer0 = layer0_sources(
                segment.memory_objs_by_layer[0], self.num_layers)
            segment.layer0_sources = layer0
        owner_keys = [
            chunk_key(keys.layer0() if isinstance(keys, _LazyLayerKeys) else keys[0])
            for keys in segment.keys_by_chunk
        ]
        cache[req_id] = dict(
            sig=sig, owners=list(owners), owner_keys=owner_keys,
            columns=segment.memory_objs_by_layer,
            starts=segment.starts, ends=segment.ends,
            keys_by_chunk=segment.keys_by_chunk, location=segment.location,
            layer0=layer0, last=stats["calls"], pos=None,
        )
        max_reqs = int(os.environ.get("SPECAGENT_VERIFY_PLAN_CACHE_MAX_REQS", "16"))
        while len(cache) > max(1, max_reqs):
            cache.popitem(last=False)
            stats["evict"] += 1
        return SpecAgentPreparedLayerwiseBatch([segment])

    def _specagent_plan_cache_drop(self, req_id: str) -> None:
        """Remove a verify plan cache entry and release its owner references."""
        cache = getattr(self, "_specagent_verify_plan_cache", None)
        if not cache:
            return
        entry = cache.pop(req_id, None)
        if entry is not None:
            for owner in entry["owners"]:
                owner.ref_count_down()

    def prepare_sparse_layerwise_plans_progressive(
        self,
        plans: list[list[tuple[int, int, CacheEngineKey]]],
        batch: SpecAgentProgressiveLayerwiseBatch,
    ) -> None:
        """Resolve and publish CPU MemoryObjs one layer at a time.

        Metadata becomes claimable immediately after the bulk membership
        lookup.  Layer 0 is then exposed as soon as it is resolved while the
        same CPU worker continues filling later layers, allowing that Python
        work to overlap model execution.
        """
        segments: list[SpecAgentProgressiveLayerwiseSegment] = []
        try:
            validated = self.bulk_validate_sparse_layerwise_plan(plans)
            if validated is None:
                batch.publish_unavailable()
                return
            starts, ends, keys_by_chunk, location = validated
            backend = self.storage_manager.storage_backends.get(location)
            if backend is None or not hasattr(backend, "batched_get_blocking"):
                batch.publish_unavailable()
                return

            offset = 0
            for plan in plans:
                next_offset = offset + len(plan)
                segments.append(
                    SpecAgentProgressiveLayerwiseSegment(
                        starts=starts[offset:next_offset],
                        ends=ends[offset:next_offset],
                        keys_by_chunk=keys_by_chunk[offset:next_offset],
                        location=location,
                        num_layers=self.num_layers,
                    )
                )
                offset = next_offset
            batch.publish_segments(segments)

            keys_layer_major = [
                list(row) for row in zip(*keys_by_chunk, strict=False)
            ]
            for layer_id, keys_layer in enumerate(keys_layer_major):
                raw_layer = backend.batched_get_blocking(keys_layer)
                acquired = [obj for obj in raw_layer if obj is not None]
                if len(acquired) != len(keys_layer):
                    for mem_obj in acquired:
                        mem_obj.ref_count_down()
                    raise RuntimeError(
                        "[SpecAgent][PROGRESSIVE-PREPARE] CPU MemoryObj "
                        f"missing at layer={layer_id}: evicted between "
                        "contains and resolve"
                    )
                offset = 0
                for segment, plan in zip(segments, plans, strict=True):
                    next_offset = offset + len(plan)
                    segment.publish(
                        layer_id, acquired[offset:next_offset]
                    )
                    offset = next_offset
                builtins._specagent_progressive_layers_published = (
                    getattr(
                        builtins,
                        "_specagent_progressive_layers_published",
                        0,
                    )
                    + 1
                )
        except BaseException as error:
            # Claimed segments are no longer present in the batch, so notify
            # every producer-owned segment directly before propagating the
            # failure to any consumers still waiting to claim.
            for segment in segments:
                segment.fail(error)
            batch.fail(error)
            raise

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def retrieve_layer(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Generator[Optional[torch.Tensor], None, None]:
        """
        Retrieve the KV cache in a layerwise manner.

        :param torch.Tensor tokens: The tokens of the corresponding KV caches.

        :param Optional[torch.Tensor] mask: The mask for the tokens. Should
            have the same length as tokens. And the mask should ALWAYS be like
            FFFFFTTTTTTT, where True means the tokens needs to be matched.

        :param **kwargs: The additional arguments for the storage backend which
            will be passed into the gpu_connector.

        return: A generator that yields Optional[torch.Tensor]. The tensor will
            be the boolean mask indicating which tokens are retrieved and will
            only be returned in the last iteration. In the first iteration,
            the generator retrieve the memory objects of the first layer from
            the storage backends. In the next iterations, it moves the KV cache
            of layer i from the memory objects (on CPU) to GPU and retrieves
            the memory objects of layer i+1 from the storage backends. In the
            last iteration, it moves the memory objects of the last layer to
            the GPU.
        """
        detailed_nvtx = bool(
            os.environ.get("SPECAGENT_DETAILED_RETRIEVE_NVTX")
        )
        detail_record = _lmc_record if detailed_nvtx else nullcontext

        # Health check: block operation if LMCache is unhealthy
        with detail_record(
            "lmcache retrieve_layer detail: health_check "
            f"tokens={len(tokens)}"
        ):
            healthy = self.is_healthy()
        if not healthy:
            logger.warning("LMCache is unhealthy, skipping retrieve_layer operation")
            yield torch.zeros(len(tokens), dtype=torch.bool)
            return

        assert self.storage_manager is not None
        assert self.gpu_connector is not None, (
            "gpu_connector is required for retrieve_layer operation"
        )

        # Get req_id for logging
        req_id = self._get_req_id(kwargs)

        specagent_expected_tokens = kwargs.get("specagent_expected_tokens")
        specagent_prefix_tokens = kwargs.get("specagent_prefix_tokens")
        specagent_chunk_plan = kwargs.get("specagent_chunk_plan")
        specagent_prepared_plan_future = kwargs.get(
            "specagent_prepared_plan_future"
        )
        specagent_prepared_segment_index = int(
            kwargs.get("specagent_prepared_segment_index", 0)
        )
        specagent_fast_path = (
            specagent_expected_tokens is not None
            and specagent_prefix_tokens is not None
        )

        with detail_record(
            "lmcache retrieve_layer detail: mask_count "
            f"req={req_id} tokens={len(tokens)}"
        ):
            if specagent_fast_path:
                num_required_tokens = int(specagent_expected_tokens)
            elif mask is not None:
                num_required_tokens = torch.sum(mask).item()
            else:
                num_required_tokens = len(tokens)
        with detail_record(
            "lmcache retrieve_layer detail: stats_begin "
            f"req={req_id} required={int(num_required_tokens)}"
        ):
            monitor_req_id = self.stats_monitor.on_retrieve_request(
                num_required_tokens
            )

        ret_mask = None
        if not specagent_fast_path:
            with detail_record(
                "lmcache retrieve_layer detail: ret_mask_alloc "
                f"req={req_id} tokens={len(tokens)}"
            ):
                ret_mask = torch.zeros(
                    len(tokens), dtype=torch.bool, device="cpu"
                )

        starts = []
        ends = []
        keys = []

        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        location = None
        retrieved_tokens_count = 0
        prepared_segment: Optional[
            Union[
                SpecAgentPreparedLayerwiseSegment,
                SpecAgentProgressiveLayerwiseSegment,
            ]
        ] = None
        with _lmc_record(
            "lmcache retrieve_layer: token_lookup "
            f"req={req_id} tokens={len(tokens)} required={int(num_required_tokens)}"
        ):
            if specagent_prepared_plan_future is not None:
                with _lmc_record(
                    "lmcache retrieve_layer: await_cpu_prepare "
                    f"req={req_id} seg={specagent_prepared_segment_index}"
                ):
                    if isinstance(
                        specagent_prepared_plan_future,
                        SpecAgentProgressiveLayerwiseBatch,
                    ):
                        prepared_batch = specagent_prepared_plan_future
                    else:
                        prepared_batch = specagent_prepared_plan_future.result()
                if prepared_batch is not None:
                    prepared_segment = prepared_batch.claim(
                        specagent_prepared_segment_index
                    )
            if getattr(self, "native_bulk_spec", False) and prepared_segment is None:
                plan = specagent_chunk_plan
                if plan is None:
                    plan = list(self.token_database.process_tokens(
                        tokens=tokens, mask=mask, request_configs=request_configs,
                        mask_num_falses=int(specagent_prefix_tokens) if specagent_fast_path else None,
                    ))
                if (os.environ.get("SPECAGENT_VERIFY_PLAN_CACHE") == "1"
                        and os.environ.get("SPECAGENT_VERIFY_SCHED_FAST") == "1"):
                    prepared = self._specagent_prepare_with_weak_plan_cache(
                        str(req_id), plan
                    )
                elif os.environ.get("SPECAGENT_VERIFY_PLAN_CACHE") == "1":
                    prepared = self._specagent_prepare_with_plan_cache(
                        str(req_id), plan
                    )
                else:
                    prepared = self.prepare_sparse_layerwise_plans([plan])
                if prepared is None:
                    raise RuntimeError("Native bulk SPEC CPU prefix is incomplete")
                prepared_segment = prepared.claim(0)
            if specagent_chunk_plan is not None:
                token_results = specagent_chunk_plan
            else:
                token_results = self.token_database.process_tokens(
                    tokens=tokens,
                    mask=mask,
                    request_configs=request_configs,
                    mask_num_falses=(
                        int(specagent_prefix_tokens)
                        if specagent_fast_path else None
                    ),
                )
            if detailed_nvtx:
                # Diagnostic-only eager materialization separates prefix hash
                # generation from backend membership checks.  The normal path
                # below remains lazy and preserves its early stop on a miss.
                with detail_record(
                    "lmcache retrieve_layer detail: token_plan_materialize "
                    f"req={req_id} tokens={len(tokens)}"
                ):
                    token_iter = iter(list(token_results))
            else:
                token_iter = iter(token_results)

            contains_context = detail_record(
                "lmcache retrieve_layer detail: storage_contains_scan "
                f"req={req_id} required={int(num_required_tokens)}"
            )
            with contains_context:
                bulk_plan = None
                if prepared_segment is not None:
                    starts = prepared_segment.starts
                    ends = prepared_segment.ends
                    keys = prepared_segment.keys_by_chunk
                    location = prepared_segment.location
                    retrieved_tokens_count = sum(
                        end - start
                        for start, end in zip(starts, ends, strict=True)
                    )
                    # Non-None sentinel prevents the synchronous membership
                    # scan; the worker already validated the entire batch.
                    bulk_plan = (starts, ends, keys, location)
                elif specagent_chunk_plan is not None:
                    with detail_record(
                        "lmcache retrieve_layer detail: bulk_contains "
                        f"req={req_id} chunks={len(specagent_chunk_plan)}"
                    ):
                        bulk_plan = self.bulk_validate_sparse_layerwise_plan(
                            [specagent_chunk_plan]
                        )
                if bulk_plan is not None:
                    starts, ends, keys, location = bulk_plan
                    retrieved_tokens_count = sum(
                        end - start for start, end in zip(
                            starts, ends, strict=True
                        )
                    )
                if bulk_plan is not None and ret_mask is not None:
                    for start, end in zip(starts, ends, strict=True):
                        ret_mask[start:end] = True
                while bulk_plan is None:
                    # Keep the aggregate token_lookup range above, but avoid
                    # three NVTX push/pop pairs per cache chunk.  At 140
                    # chunks those profiling calls alone inflated a 4.4 ms
                    # lookup to 133 ms in Nsight and produced artificial
                    # 200 ms forward outliers.
                    try:
                        start, end, key = next(token_iter)
                    except StopIteration:
                        break
                    assert isinstance(key, CacheEngineKey)

                    keys_multi_layer = key.split_layers(self.num_layers)

                    hit_layers, block_mapping = (
                        self.storage_manager.batched_contains(
                            keys_multi_layer,
                            self.retrieve_locations,
                        )
                    )
                    if (
                        hit_layers == self.num_layers
                        and len(block_mapping) == 1
                    ):
                        current_location = next(iter(block_mapping.keys()))
                        if location is None:
                            location = current_location
                        else:
                            # TODO(Jiayi): Support multi-location retrieval
                            # in the future.
                            assert current_location == location, (
                                "All retrieved keys should be from the same "
                                "location when use layerwise retrieval."
                                "Please support multi-location retrieval in "
                                "the future."
                            )
                    else:
                        break

                    starts.append(start)
                    ends.append(end)
                    keys.append(keys_multi_layer)

                    if ret_mask is not None:
                        ret_mask[start:end] = True
                    retrieved_tokens_count += end - start

        if keys:
            # Transpose the keys into layer major format
            with _lmc_record(
                "lmcache retrieve_layer: setup_layerwise "
                f"req={req_id} chunks={len(keys)} "
                f"tokens={retrieved_tokens_count}"
            ):
                with _lmc_record(
                    "lmcache retrieve_layer: setup transpose_keys "
                    f"req={req_id} chunks={len(keys)}"
                ):
                    # Consumed only without a prepared segment or by the
                    # non-native full-plan branch; building it eagerly would
                    # split every lazily kept native chunk key.
                    keys_layer_major = None

                    def _keys_layer_major():
                        return [list(row) for row in zip(*keys, strict=False)]

                with _lmc_record(
                    "lmcache retrieve_layer: setup get_generator "
                    f"req={req_id} chunks={len(keys)}"
                ):
                    get_generator = None
                    if prepared_segment is None:
                        get_generator = (
                            self.storage_manager.layerwise_batched_get(
                                _keys_layer_major(),
                                location=location,
                            )
                        )

                with _lmc_record(
                    "lmcache retrieve_layer: setup init_gpu_consumer "
                    f"req={req_id} chunks={len(keys)}"
                ):
                    assert_layerwise_gpu_connector(self.gpu_connector)

                    _sa_l0 = getattr(prepared_segment, "layer0_sources", None)
                    if _sa_l0 is not None:
                        kwargs["specagent_layer0_sources"] = _sa_l0
                    mem_obj_consumer = self.gpu_connector.batched_to_gpu(
                        starts, ends, **kwargs
                    )
                    next(mem_obj_consumer)

            to_count_down = []
            deferred_source_owner = kwargs.get(
                "specagent_deferred_source_owner"
            )
            if isinstance(deferred_source_owner, dict):
                deferred_source_owner.setdefault("memory_objs", [])

            def hold_source_objs(memory_objs: list[Any]) -> None:
                if isinstance(deferred_source_owner, dict):
                    deferred_source_owner["memory_objs"].extend(memory_objs)
                else:
                    to_count_down.extend(memory_objs)
            progressive_segment = (
                prepared_segment
                if isinstance(
                    prepared_segment, SpecAgentProgressiveLayerwiseSegment
                )
                else None
            )
            full_plan_objs = (
                prepared_segment.memory_objs_by_layer
                if isinstance(
                    prepared_segment, SpecAgentPreparedLayerwiseSegment
                )
                else None
            )
            if full_plan_objs is not None:
                native_owners = prepared_segment.source_owners
                if native_owners is not None and isinstance(deferred_source_owner, dict):
                    # The whole-plan CUDA owner replaces per-layer leases.
                    # Borrowed tensor descriptors carry no references.
                    hold_source_objs(native_owners)
                    prepared_segment.source_owners = []
                else:
                    if native_owners is not None:
                        # Consumers without a whole-plan owner still receive
                        # independently leased layer readers.
                        try:
                            full_plan_objs = acquire_bulk_layer_views(
                                native_owners, self.num_layers
                            )
                        finally:
                            for owner in native_owners:
                                owner.ref_count_down()
                            prepared_segment.source_owners = []
                    for mem_objs_layer in full_plan_objs:
                        hold_source_objs(mem_objs_layer)
            elif kwargs.get("specagent_full_plan_fetch"):
                resolve_location = location or "LocalCPUBackend"
                backend = self.storage_manager.storage_backends.get(
                    resolve_location
                )
                if backend is not None and hasattr(
                    backend, "batched_get_blocking"
                ):
                    # Resolve every layer's CPU MemoryObj references here, in
                    # the retriever's first step, so the per-layer loop below
                    # never touches the get executor, futures, or
                    # task.result().  References are held (ref_count_up inside
                    # batched_get_blocking) and released via to_count_down
                    # exactly like the task path.
                    with _lmc_record(
                        "lmcache retrieve_layer: full_plan_resolve "
                        f"req={req_id} layers={self.num_layers} "
                        f"chunks={len(keys)}"
                    ):
                        full_plan_objs = [
                            backend.batched_get_blocking(keys_layer)
                            for keys_layer in _keys_layer_major()
                        ]
                    for layer_id, mem_objs_layer in enumerate(full_plan_objs):
                        for chunk_idx, mem_obj in enumerate(mem_objs_layer):
                            if mem_obj is None:
                                raise RuntimeError(
                                    "[SpecAgent][FULL-PLAN] CPU MemoryObj "
                                    f"missing for req={req_id} "
                                    f"layer={layer_id} chunk={chunk_idx}: "
                                    "evicted between contains and resolve"
                                )
                        to_count_down.extend(mem_objs_layer)

            if full_plan_objs is not None or progressive_segment is not None:
                progressive_complete = False
                try:
                    for layer_id in range(self.num_layers):
                        if layer_id == 0:
                            if specagent_expected_tokens is not None:
                                yield retrieved_tokens_count
                            else:
                                assert ret_mask is not None
                                yield torch.sum(ret_mask)
                        else:
                            yield None
                        if progressive_segment is not None:
                            with _lmc_record(
                                "lmcache retrieve_layer: wait_progressive_layer "
                                f"req={req_id} layer={layer_id} "
                                f"chunks={len(keys)}"
                            ):
                                mem_objs_layer = (
                                    progressive_segment.take_layer(layer_id)
                                )
                            hold_source_objs(mem_objs_layer)
                        else:
                            assert full_plan_objs is not None
                            mem_objs_layer = full_plan_objs[layer_id]
                        with _lmc_record(
                            "lmcache retrieve_layer: gpu_send "
                            f"req={req_id} layer={layer_id} chunks={len(keys)}"
                        ):
                            mem_obj_consumer.send(mem_objs_layer)
                    progressive_complete = True
                finally:
                    if progressive_segment is not None:
                        progressive_segment.release_pending()
                        if not progressive_complete:
                            for mem_obj in to_count_down:
                                mem_obj.ref_count_down()
                            to_count_down.clear()
            else:
                assert get_generator is not None
                cpu_prefetch_ahead = max(
                    0,
                    int(
                        kwargs.get(
                            "specagent_cpu_prefetch_ahead_layers",
                            os.environ.get(
                                "SPECAGENT_CPU_PREFETCH_AHEAD_LAYERS", "0"
                            ),
                        )
                        or 0
                    ),
                )
                cpu_prefetch_profile = bool(
                    os.environ.get("SPECAGENT_CPU_PREFETCH_PROFILE")
                )
                prefetch_ms_total = 0.0
                result_ms_total = 0.0
                send_ms_total = 0.0
                result_ms_max = 0.0
                send_ms_max = 0.0
                prefetched_tasks: dict[int, Any] = {}
                next_task_layer = 0

                def prefetch_until(exclusive_layer: int) -> None:
                    nonlocal next_task_layer, prefetch_ms_total
                    while next_task_layer < min(
                        self.num_layers, exclusive_layer
                    ):
                        prefetch_t0 = time.perf_counter()
                        with _lmc_record(
                            "lmcache retrieve_layer: submit_get "
                            f"req={req_id} layer={next_task_layer} "
                            f"chunks={len(keys)}"
                        ):
                            prefetched_tasks[next_task_layer] = next(
                                get_generator
                            )
                        prefetch_ms_total += (
                            time.perf_counter() - prefetch_t0
                        ) * 1000.0
                        assert prefetched_tasks[next_task_layer] is not None
                        next_task_layer += 1

                for layer_id in range(self.num_layers):
                    prefetch_until(layer_id + cpu_prefetch_ahead + 1)
                    task = prefetched_tasks.pop(layer_id)

                    if layer_id == 0:
                        # NOTE(Yuwei): For sglang integration we need to
                        # provide retrieved tokens number in the first layer
                        # loading since there is no lookup
                        if specagent_expected_tokens is not None:
                            yield retrieved_tokens_count
                        else:
                            assert ret_mask is not None
                            yield torch.sum(ret_mask)
                    else:
                        yield None

                    result_t0 = time.perf_counter()
                    with _lmc_record(
                        "lmcache retrieve_layer: task_result "
                        f"req={req_id} layer={layer_id} chunks={len(keys)}"
                    ):
                        mem_objs_layer = task.result()
                    result_ms = (time.perf_counter() - result_t0) * 1000.0
                    result_ms_total += result_ms
                    result_ms_max = max(result_ms_max, result_ms)
                    send_t0 = time.perf_counter()
                    with _lmc_record(
                        "lmcache retrieve_layer: gpu_send "
                        f"req={req_id} layer={layer_id} chunks={len(keys)}"
                    ):
                        mem_obj_consumer.send(mem_objs_layer)
                    send_ms = (time.perf_counter() - send_t0) * 1000.0
                    send_ms_total += send_ms
                    send_ms_max = max(send_ms_max, send_ms)
                    hold_source_objs(mem_objs_layer)

                if cpu_prefetch_profile:
                    logger.info(
                        "[SpecAgent][CPU-PREFETCH] req=%s layers=%d ahead=%d "
                        "chunks=%d tokens=%d prefetch_ms=%.3f "
                        "task_result_ms_sum=%.3f task_result_ms_max=%.3f "
                        "gpu_send_ms_sum=%.3f gpu_send_ms_max=%.3f",
                        req_id,
                        self.num_layers,
                        cpu_prefetch_ahead,
                        len(keys),
                        retrieved_tokens_count,
                        prefetch_ms_total,
                        result_ms_total,
                        result_ms_max,
                        send_ms_total,
                        send_ms_max,
                    )

            for mem_obj in to_count_down:
                mem_obj.ref_count_down()
        else:
            # If no cache are found, we still need to yield to avoid
            # `StopIteration`
            for layer_id in range(self.num_layers):
                if layer_id == 0 and specagent_expected_tokens is not None:
                    yield 0
                else:
                    yield None

            yield 0 if specagent_expected_tokens is not None else None
            retrieved_tokens = retrieved_tokens_count
            if ret_mask is not None:
                retrieved_tokens = torch.sum(ret_mask)
            self.stats_monitor.on_retrieve_finished(monitor_req_id,
                                                    retrieved_tokens)
            if not self._is_passive():
                logger.info(
                    "[req_id=%s] Retrieved %d out of %d out of total %d tokens",
                    req_id,
                    retrieved_tokens,
                    num_required_tokens,
                    len(tokens),
                )
            return

        yield None

        # synchronize the last layer
        next(mem_obj_consumer)

        # Do not unpin based on the object's ambient pin state.  Layerwise
        # batched_get owns the references released above, but it does not
        # acquire a pin.  A visible pin commonly belongs to the scheduler's
        # lookup(pin=True) and must be released exactly once by
        # lookup_unpin(req_id).  Treating is_pinned as ownership caused every
        # retrieved LocalCPU key to be unpinned here and then again at the
        # request boundary.

        retrieved_tokens = retrieved_tokens_count
        if ret_mask is not None:
            retrieved_tokens = torch.sum(ret_mask)
        self.stats_monitor.on_retrieve_finished(monitor_req_id, retrieved_tokens)
        if not self._is_passive():
            logger.info(
                "[req_id=%s] Retrieved %d out of %d out of total %d tokens",
                req_id,
                retrieved_tokens,
                num_required_tokens,
                len(tokens),
            )

        yield retrieved_tokens if specagent_fast_path else ret_mask

    @_lmcache_nvtx_annotate
    def lookup(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        search_range: Optional[List[str]] = None,
        lookup_id: Optional[str] = None,
        pin: bool = False,
        request_configs: Optional[dict] = None,
    ) -> int:
        """
        Checks the existence of KV cache of the tokens from the cache engine.

        :param Optional[Union[torch.Tensor, List[int]]] tokens: the input tokens,
        with shape [seq_len]

        :param Optional[List[int]] hashes: the input hashes, with length [num_chunks]
        :param Optional[List[int]] offsets: the offsets of each chunk,
        with length [num_chunks]

        :param Optional[List[str]] search_range: The range of storage backends
        to search in. Should be a subset of
        ["LocalCPUBackend", "LocalDiskBackend"] for now.
        If None, search in all backends.

        :param Optional[str] lookup_id: The lookup ID to
            associate with the lookup. When pin is true, this argument is
            required to be not None.

        :param bool pin: If True, pin the KV cache in the storage.

        :param Optional[dict] request_configs: the configs of the request.

        :return: An int indicating how many prefix tokens exist inside LMCache.
        """
        # Health check: block operation if LMCache is unhealthy
        if not self.is_healthy():
            logger.warning("LMCache is unhealthy, skipping lookup operation")
            return 0

        assert self.storage_manager is not None

        if tokens is not None:
            lookup_stats = self.stats_monitor.on_lookup_request(len(tokens))
        else:
            assert offsets is not None
            assert hashes is not None
            lookup_stats = self.stats_monitor.on_lookup_request(sum(offsets))

        if search_range is None:
            search_range = self.retrieve_locations

        res = 0
        try:
            chunk_info_iterator = self.token_database.process_tokens(
                tokens=tokens,
                hashes=hashes,
                offsets=offsets,
                request_configs=request_configs,
            )

            # TODO: support batched_contains when layerwise is enabled
            if self.use_layerwise:
                _dbg = os.environ.get("LMCACHE_LOOKUP_DEBUG")
                _ci = 0
                for start, end, key in chunk_info_iterator:
                    assert isinstance(key, CacheEngineKey)

                    # TODO(Jiayi): Optimize by checking only the existence of the key
                    # of one layer
                    key_all_layers = [key] if getattr(self, "grouped_layers", False) else key.split_layers(self.num_layers)

                    hit_chunks, block_mapping = self.storage_manager.batched_contains(
                        key_all_layers,  # type: ignore
                        search_range,
                        pin,
                    )
                    # Only all layers are hit and hit in one location,
                    # we consider this key as a hit
                    if hit_chunks == len(key_all_layers) and len(block_mapping) == 1:
                        if pin:
                            assert lookup_id is not None, (
                                "lookup_id is required when pin is True"
                            )
                            location = next(iter(block_mapping.keys()))
                            self.lookup_pins[lookup_id][location].extend(key_all_layers)
                        res = end
                        _ci += 1
                        continue
                    if pin:
                        for location, pinned_keys in block_mapping.items():
                            self.storage_manager.batched_unpin(pinned_keys, [location])
                    # FIRST MISS: log where in the prefix the lookup stops and why,
                    # then return res (LMCache stops at first unavailable chunk).
                    if _dbg:
                        try:
                            import builtins as _b
                            _fh = getattr(_b, "_lmc_lookupdbg_fh", None)
                            if _fh is None:
                                _fh = _b._lmc_lookupdbg_fh = open(_dbg, "a", buffering=1)
                                _b._lmc_lookupdbg_n = 0
                            if _b._lmc_lookupdbg_n < 8000:
                                _b._lmc_lookupdbg_n += 1
                                if tokens is not None:
                                    _tk = tokens[start:end]
                                    if hasattr(_tk, "tolist"):
                                        _tk = _tk.tolist()
                                    _fp = "n%d_f%d_l%d_s%d" % (
                                        len(_tk), _tk[0] if _tk else 0,
                                        _tk[-1] if _tk else 0, sum(_tk) & 0xffffff)
                                else:
                                    _fp = "NOTOKENS_hashes"
                                _fh.write(
                                    "MISS ts=%.4f id=%s miss_chunk=%d "
                                    "miss_span=[%d:%d] res=%d hit_layers=%d/%d "
                                    "locs=%d tokfp=%s key0=%s\n" % (
                                        time.monotonic(),
                                        lookup_id, _ci, start, end, res, hit_chunks,
                                        self.num_layers, len(block_mapping),
                                        _fp,
                                        key_all_layers[0].to_string()[:80],
                                    )
                                )
                        except Exception:
                            pass
                    return res
                # whole prefix hit (loop exhausted) — log full hit occasionally
                if _dbg:
                    try:
                        import builtins as _b
                        _fh = getattr(_b, "_lmc_lookupdbg_fh", None)
                        if _fh is None:
                            _fh = _b._lmc_lookupdbg_fh = open(_dbg, "a", buffering=1)
                            _b._lmc_lookupdbg_n = 0
                        if _b._lmc_lookupdbg_n < 8000:
                            _b._lmc_lookupdbg_n += 1
                            _fh.write("FULLHIT id=%s chunks=%d res=%d\n"
                                      % (lookup_id, _ci, res))
                    except Exception:
                        pass
            else:
                chunk_info_list = []
                keys = []
                for chunk_info in chunk_info_iterator:
                    assert isinstance(chunk_info[2], CacheEngineKey)
                    start, end, _ = chunk_info
                    chunk_info_list.append(chunk_info)
                    # chunk_info contains (start, end, key)
                    # chunk_info[2] is the key
                    keys.append(chunk_info[2])
                # hit chunks by prefix matching
                hit_chunks, block_mapping = self.storage_manager.batched_contains(
                    keys, search_range, pin
                )
                if pin and block_mapping:
                    assert lookup_id is not None, (
                        "lookup_id is required when pin is True"
                    )
                    self.lookup_pins[lookup_id] = block_mapping
                for idx, (start, end, key) in enumerate(chunk_info_list):
                    if idx < hit_chunks:
                        res = end
                        continue
                    return res

            # all tokens where found, return the maximal end
            return res
        finally:
            self.stats_monitor.on_lookup_finished(lookup_stats, res)
            # vllm lookup sets pin to True
            if pin:
                # touch_cache is tightly coupled with batched_contains
                self.storage_manager.touch_cache()

    @_lmcache_nvtx_annotate
    def move(
        self,
        tokens: Union[torch.Tensor, List[int]],
        old_position: str,
        new_position: tuple[str, str],
        event_id: str,
        do_copy: bool = True,
    ) -> int:
        """
        Perform cross-node move of the KV cache.
        """
        assert self.storage_manager is not None

        num_tokens = self.lookup(
            tokens,
            search_range=[old_position],
            lookup_id=event_id,
            pin=True,
        )

        if not num_tokens:
            logger.debug("Move is not performed as there are no tokens to move.")
            return 0

        block_mapping = self.lookup_pins[event_id]
        assert len(block_mapping) == 1
        keys = block_mapping[old_position]

        memory_objs = self.storage_manager.batched_get(
            keys=keys,
            location=old_position,
        )
        assert None not in memory_objs, "Failed to get memory objects to move"
        logger.debug(
            f"Trying to send {len(memory_objs)} memory objects to {new_position}"
        )

        # TODO: reduce loops
        token_dim = memory_objs[0].meta.fmt.token_dim()  # type: ignore
        offsets = [m.meta.shape[token_dim] for m in memory_objs]  # type: ignore

        transfer_spec = {
            "target_peer_init_url": new_position[0],
            "offsets": offsets,
        }

        logger.info(self.storage_manager.storage_backends)
        p2p_backend = self.storage_manager.storage_backends["P2PBackend"]

        future = asyncio.run_coroutine_threadsafe(
            p2p_backend.async_batched_submit_put_task(
                keys,
                memory_objs,  # type: ignore
                transfer_spec=transfer_spec,
            ),
            self.storage_manager.loop,
        )

        future.result()

        if not do_copy:
            self.storage_manager.batched_remove(keys, locations=[old_position])

        logger.debug(f"Moving {num_tokens} token from {old_position} to {new_position}")
        return num_tokens

    # TODO(Jiayi): Add layerwise support.
    @_lmcache_nvtx_annotate
    def async_lookup_and_prefetch(
        self,
        lookup_id: str,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        hashes: Optional[List[int]] = None,
        offsets: Optional[List[int]] = None,
        search_range: Optional[List[str]] = None,
        pin: bool = False,
        request_configs: Optional[dict] = None,
    ) -> None:
        """
        An async version of lookup + prefetch.

        There are three categories of backends:
        (1) sync lookup + sync retrieval (e.g., cpu)
        (2) sync lookup + async retrieval (e.g., disk)
        (3) async lookup + async retrieval (e.g., p2p)
        """
        assert self.storage_manager is not None

        keys: list[CacheEngineKey] = []
        cum_chunk_lengths = [0]

        if search_range is None:
            search_range = self.retrieve_locations

        # TODO(Jiayi): make token database able to return list.
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens,
            hashes=hashes,
            offsets=offsets,
            request_configs=request_configs,
        ):
            assert isinstance(key, CacheEngineKey)
            keys.append(key)
            cum_chunk_lengths.append(end)

        asyncio.run_coroutine_threadsafe(
            self.storage_manager.async_lookup_and_prefetch(
                lookup_id, keys, cum_chunk_lengths, search_range, pin
            ),
            self.storage_manager.loop,
        )

    def cleanup_memory_objs(self, lookup_id: str) -> None:
        """
        Cleanup memory objects allocated during prefetch for an aborted lookup.

        Called by the scheduler when it determines that an aborted lookup
        has finished its prefetch tasks.
        """
        try:
            # Get the completed future from event_manager
            if (
                self.event_manager.get_event_status(EventType.LOADING, lookup_id)
                != EventStatus.DONE
            ):
                logger.debug(
                    "No completed event found for lookup_id=%s to clean up.", lookup_id
                )
                return
            future = self.event_manager.pop_event(EventType.LOADING, lookup_id)

            # Get memory objects from the future result
            memory_objs = future.result()
            # Flatten nested lists (each backend returns a list of chunks)
            memory_objs_flat = [mm for m in memory_objs for mm in m]

            # Release each memory object
            for key, memory_obj in memory_objs_flat:
                try:
                    logger.debug("Releasing memory object for lookup_id=%s", lookup_id)
                    memory_obj.unpin()
                    memory_obj.ref_count_down()
                except Exception as e:
                    logger.error(f"Error releasing memory object: {e}")
        except Exception as e:
            logger.error(
                f"Error during cleanup_memory_objs for lookup_id={lookup_id}: {e}"
            )

    # TODO(Jiayi): Need to handle the case where `tokens=None`.
    # In this case, we compress all tokens.
    # TODO(Jiayi): support other compression methods.
    @_lmcache_nvtx_annotate
    def compress(
        self,
        tokens: Union[torch.Tensor, List[int]],
        method: str,
        location: str,
        event_id: str,
    ) -> int:
        assert self.storage_manager is not None
        if method not in ["cachegen"]:
            logger.warning(f"Unsupported compression method: {method}.")
            return 0

        # First Party
        from lmcache.v1.storage_backend.naive_serde import CreateSerde

        serializer, _ = CreateSerde(method, self.metadata, self.config)

        num_tokens = self.lookup(
            tokens,
            search_range=[location],
            lookup_id=event_id,
            pin=True,
        )

        if not num_tokens:
            logger.debug("Move is not performed as there are no tokens to move.")
            return 0

        block_mapping = self.lookup_pins[event_id]
        assert len(block_mapping) == 1
        keys = block_mapping[location]

        memory_objs = self.storage_manager.batched_get(
            keys=keys,
            location=location,
        )
        assert None not in memory_objs, (
            "LMCacheEngine.compress: Failed to get memory objects to compress"
        )

        compressed_memory_objs = []
        for memory_obj in memory_objs:
            assert memory_obj is not None
            compressed_memory_obj = serializer.serialize(memory_obj)
            memory_obj.unpin()
            compressed_memory_objs.append(compressed_memory_obj)

        self.storage_manager.batched_remove(keys, locations=[location])

        self.storage_manager.batched_put(
            keys=keys,
            memory_objs=compressed_memory_objs,
            location=location,
        )

        return num_tokens

    @_lmcache_nvtx_annotate
    def decompress(
        self,
        tokens: Union[torch.Tensor, List[int]],
        method: str,
        location: str,
        event_id: str,
    ) -> int:
        assert self.storage_manager is not None
        if method not in ["cachegen"]:
            logger.warning(f"Unsupported decompression method: {method}.")
            return 0

        # First Party
        from lmcache.v1.storage_backend.naive_serde import CreateSerde

        _, deserializer = CreateSerde(method, self.metadata, self.config)

        num_tokens = self.lookup(
            tokens,
            search_range=[location],
            lookup_id=event_id,
            pin=True,
        )

        if not num_tokens:
            logger.debug("there are no tokens to decompress.")
            return 0

        block_mapping = self.lookup_pins[event_id]
        assert len(block_mapping) == 1
        keys = block_mapping[location]

        compressed_memory_objs = self.storage_manager.batched_get(
            keys=keys,
            location=location,
        )

        assert None not in compressed_memory_objs, (
            "LMCacheEngine.compress: Failed to get compressed "
            "memory objects to decompress"
        )

        memory_objs = []
        for compressed_memory_obj in compressed_memory_objs:
            assert compressed_memory_obj is not None
            memory_obj = deserializer.deserialize(compressed_memory_obj)
            compressed_memory_obj.unpin()
            memory_objs.append(memory_obj)

        self.storage_manager.batched_remove(keys, locations=[location])

        self.storage_manager.batched_put(
            keys=keys,
            memory_objs=memory_objs,
            location=location,
        )

        return num_tokens

    @_lmcache_nvtx_annotate
    def lookup_unpin(self, lookup_id: str) -> None:
        if lookup_id in self.lookup_pins:
            assert self.storage_manager is not None
            for location, keys in self.lookup_pins.pop(lookup_id).items():
                self.storage_manager.batched_unpin(keys, [location])

        elif (
            self.async_loading is not None
            and self.event_manager.get_event_status(EventType.LOADING, lookup_id)
            != EventStatus.NOT_FOUND
        ):
            self.cleanup_memory_objs(lookup_id)

    @_lmcache_nvtx_annotate
    def clear(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        locations: Optional[List[str]] = None,
        request_configs: Optional[dict] = None,
    ) -> int:
        # TODO: need to clear by request_configs
        if self.save_only_first_rank:
            if self.metadata.is_first_rank():
                num_removed = self._clear(tokens, locations, request_configs)
                return num_removed
            else:
                return 0
        return self._clear(tokens, locations, request_configs)

    @_lmcache_nvtx_annotate
    def get_kv_events(self) -> Iterable[CacheStoreEvent]:
        if self.kv_events_enabled and (events := self.kv_events):
            self.kv_events = []
            return events
        return []

    def _clear(
        self,
        tokens: Optional[Union[torch.Tensor, List[int]]] = None,
        locations: Optional[List[str]] = None,
        request_configs: Optional[dict] = None,
    ) -> int:
        assert self.storage_manager is not None
        assert isinstance(self.storage_manager, StorageManager)
        # Clear all caches if tokens is None
        if tokens is None or len(tokens) == 0:
            num_cleared = self.storage_manager.clear(locations)
            return num_cleared

        num_removed = 0
        # Only remove the caches for the given tokens
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens, request_configs=request_configs
        ):
            assert isinstance(key, CacheEngineKey)
            removed = self.storage_manager.remove(key, locations)
            num_removed += removed
        return num_removed

    @_lmcache_nvtx_annotate
    def health(
        self,
    ) -> int:
        """
        Check the health of the cache engine.
        return: 0 if healthy, otherwise the error code
        """
        assert self.storage_manager is not None
        return 0 if self.storage_manager.memcheck() else -1

    def close(self) -> None:
        """Close the cache engine and free all the resources"""
        logger.info("Closing LMCacheEngine...")

        if self.lmcache_worker is not None:
            try:
                logger.info("Closing lmcache_worker...")
                self.lmcache_worker.close()
                logger.info("lmcache_worker closed successfully")
            except Exception as e:
                logger.error(f"Error closing lmcache_worker: {e}")

        try:
            logger.info("Closing storage_manager...")
            if self.storage_manager is not None:
                self.storage_manager.close()
            logger.info("storage_manager closed successfully")
        except Exception as e:
            logger.error(f"Error closing storage_manager: {e}")

        logger.info("LMCacheEngine closed.")

    def _async_process_tokens_internal(
        self,
        tokens,
        mask,
        ret_mask,
        **kwargs,
    ) -> ProcessTokensInternalResult:
        """
        This function is used to get the memory objects from the event manager.

        Args:
            tokens: Input tokens to process
            mask: Mask indicating valid token positions
            ret_mask: Output mask updated with cache hit positions
            **kwargs: Additional keyword arguments
        """
        assert "req_id" in kwargs, "req_id is required for async loading"
        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        tot_kv_size = 0
        chunks: List[ProcessedChunk] = []
        future = self.event_manager.get_event_future(
            EventType.LOADING, kwargs["req_id"]
        )
        # As mentioned in async_lookup_and_prefetch(), the future.result()
        # is key data pair for each chunk in each tier. So extract the key
        # and memory object pairs to memory_obj_map
        try:
            keyed_memory_objs = future.result()
            memory_obj_map: dict[CacheEngineKey, MemoryObj] = {}
        except Exception as e:
            logger.error(f"Error popping event for request {kwargs['req_id']}: {e}")
            return [], 0

        for backend_results in keyed_memory_objs:
            for key, memory_obj in backend_results:
                memory_obj_map[key] = memory_obj

        # TODO(Jiayi): hashing inside `process_tokens` can be skipped.
        used_keys: set[CacheEngineKey] = set()
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens,
            mask=mask,
            request_configs=request_configs,
        ):
            assert isinstance(key, CacheEngineKey)
            memory_obj = memory_obj_map.get(key)
            if memory_obj is None:
                # returned chunks are expected to be contiguous.
                # break at the first missing chunk.
                break
            chunks.append((key, memory_obj, start, end))
            tot_kv_size += memory_obj.get_size()
            ret_mask[start:end] = True
            used_keys.add(key)

        # NOTE: free the memory objects that are not hit.
        for key, mem_obj in memory_obj_map.items():
            if key not in used_keys:
                mem_obj.ref_count_down()

        return chunks, tot_kv_size

    def _process_tokens_internal(
        self,
        tokens,
        mask,
        ret_mask,
        **kwargs,
    ) -> ProcessTokensInternalResult:
        """Process tokens and populate the reordered lists.

        This function is used to process tokens and populate the reordered lists.

        Args:
            tokens: Input tokens to process
            mask: Mask indicating valid token positions
            ret_mask: Output mask updated with cache hit positions
            **kwargs: Additional keyword arguments
        """
        assert self.storage_manager is not None

        tot_kv_size = 0
        reordered_chunks: List[ProcessedChunk] = []
        request_configs = kwargs.get("request_configs")
        if request_configs is not None and len(request_configs) != 0:
            assert isinstance(request_configs, dict)

        chunk_infos = []
        for start, end, key in self.token_database.process_tokens(
            tokens=tokens,
            mask=mask,
            request_configs=request_configs,
        ):
            assert isinstance(key, CacheEngineKey)
            chunk_infos.append((key, start, end))

        # block_mapping: location -> [(CacheEngineKey, start, end)]
        if (
            "req_id" in kwargs
            and kwargs["req_id"] in self.lookup_pins
            and len(self.lookup_pins[kwargs["req_id"]]) == 1
        ):
            location = next(iter(self.lookup_pins[kwargs["req_id"]].keys()))
            block_mapping = {location: chunk_infos}
        else:
            block_mapping = self.storage_manager.get_block_mapping(chunk_infos)

        last_failed_block_start = None
        for location, blocks in block_mapping.items():
            keys = [key for key, _, _ in blocks]
            memory_objs = self.storage_manager.batched_get(
                keys=keys,
                location=location,
            )

            for (key, start, end), memory_obj in zip(blocks, memory_objs, strict=False):
                if memory_obj is None:
                    logger.warning(
                        "The cache block is in the storage, but it can't be retrieved"
                    )
                    if (
                        last_failed_block_start is None
                        or last_failed_block_start < start
                    ):
                        last_failed_block_start = start
                    break
                reordered_chunks.append((key, memory_obj, start, end))
                tot_kv_size += memory_obj.get_size()
                ret_mask[start:end] = True

        if last_failed_block_start is not None:
            ret_mask[last_failed_block_start:] = False

            reordered_chunks = [
                (key, memory_obj, start, end)
                for key, memory_obj, start, end in reordered_chunks
                if end < last_failed_block_start
            ]
        return reordered_chunks, tot_kv_size

    def _broadcast_or_receive_memory_objs(
        self,
        reordered_chunks,
        ret_mask,
    ):
        """
        Handles broadcasting or receiving memory objects in a distributed environment.

        This function implements the communication logic where:
        - The first rank (coordinator) broadcasts memory objects and metadata to others
        - Other ranks receive and reconstruct the memory objects

        Parameters:
        reordered_chunks: List of tuples containing [key, memory object, start, end]
        ret_mask: Boolean mask indicating which positions have been processed

        Side Effects:
        - On first rank:
          * Broadcasts chunk count and each chunk's combined metadata
          * Broadcasts tensor data
        - On other ranks:
          * Receives chunk data and populates reordered_chunks
          * Updates ret_mask to mark received positions as True
        """
        if self.metadata.is_first_rank():
            # Broadcast total chunk count
            chunk_count = len(reordered_chunks)
            self.broadcast_object_fn(chunk_count, self.metadata.first_rank)

            # Broadcast each chunk's data
            for key, memory_obj, start, end in reordered_chunks:
                # Combine (start, end) and metadata into single broadcast
                metadata_dict = memory_obj.metadata.to_dict()
                combined_metadata = (start, end, metadata_dict)
                self.broadcast_object_fn(combined_metadata, self.metadata.first_rank)

                # Broadcast tensor data
                raw_tensor = memory_obj.raw_tensor
                assert raw_tensor is not None
                tensor_to_broadcast = raw_tensor.to(f"cuda:{self.metadata.worker_id}")
                self.broadcast_fn(tensor_to_broadcast, self.metadata.first_rank)
        else:
            # Receive total chunk count
            chunk_count = self.broadcast_object_fn(None, self.metadata.first_rank)
            if chunk_count is None:
                logger.warning(
                    f"rank={self.metadata.worker_id} received None chunk_count"
                )
                return

            # Fill reordered_chunks with received data
            for _ in range(chunk_count):
                # Receive combined metadata (start, end, metadata_dict)
                combined_metadata = self.broadcast_object_fn(
                    None, self.metadata.first_rank
                )
                if combined_metadata is None:
                    logger.warning(
                        f"rank={self.metadata.worker_id} "
                        "received None combined_metadata"
                    )
                    break
                start, end, metadata_dict = combined_metadata
                ret_mask[start:end] = True

                # Create tensor and receive data
                metadata = MemoryObjMetadata.from_dict(metadata_dict)
                local_rank = self.metadata.worker_id % torch.cuda.device_count()
                raw_tensor = torch.empty(
                    torch.Size([metadata.get_size()]),
                    dtype=torch.uint8,
                    device=f"cuda:{local_rank}",
                )
                self.broadcast_fn(raw_tensor, self.metadata.first_rank)

                # Create temporary memory object (key not needed for other ranks)
                memory_obj = TensorMemoryObj(
                    raw_data=raw_tensor, metadata=metadata, parent_allocator=None
                )
                reordered_chunks.append((None, memory_obj, start, end))

    def _is_passive(self):
        """
        A 'passive' CacheEngine means that the node itself will not store/retrieve
        the data directly, but from the "active" worker (i.e., rank 0 in MLA)
        """
        return self.save_only_first_rank and not self.metadata.is_first_rank()

    def _get_slot_mapping_list(
        self,
        slot_mapping: Optional[Union[torch.Tensor, List[int]]],
    ) -> Optional[List[int]]:
        """
        Convert slot_mapping to list if it's a tensor, otherwise return as is.

        :param slot_mapping: The slot_mapping to convert,
            can be a torch.Tensor or List[int], or None
        :type slot_mapping: Optional[Union[torch.Tensor, List[int]]]
        :return: The slot_mapping as a List[int], or None if input is None
        :rtype: Optional[List[int]]
        """
        if slot_mapping is None:
            return None
        if isinstance(slot_mapping, torch.Tensor):
            return slot_mapping.tolist()
        # At this point, slot_mapping must be List[int]
        return slot_mapping

    def _log_kvcache_for_check(
        self,
        operation: str,
        kwargs: dict,
        token_count: int,
        require_req_id: bool = False,
    ) -> None:
        """
        Helper method to log KVCache Check information.

        This method centralizes the KVCache Check logging logic that was
        duplicated in multiple methods.

        Args:
            operation: The operation being performed (e.g., "Store", "retrieve")
            kwargs: The keyword arguments containing slot_mapping and req_id
            token_count: The number of tokens involved in the operation
            require_req_id: Whether req_id must be present (default: False)
        """
        if not self.kvcache_check_log_enabled:
            return

        slot_mapping = kwargs.get("slot_mapping")
        if slot_mapping is None:
            return

        if require_req_id:
            req_id = kwargs.get("req_id")
            if req_id is None:
                return
        else:
            req_id = kwargs.get("req_id", "unspecified")

        # Convert slot_mapping to list if it's a tensor
        slot_mapping_list = self._get_slot_mapping_list(slot_mapping)
        # slot_mapping_list should not be None when slot_mapping is not None
        assert slot_mapping_list is not None

        logger.info(
            "[KVCache Check] %s request %s, tokens=%d, slot_mapping: %s",
            operation,
            req_id,
            token_count,
            compress_slot_mapping(slot_mapping_list),
        )


class LMCacheEngineBuilder:
    _instances: Dict[str, LMCacheEngine] = {}
    _cfgs: Dict[str, LMCacheEngineConfig] = {}
    _metadatas: Dict[str, LMCacheMetadata] = {}
    _stat_loggers: Dict[str, LMCacheStatsLogger] = {}

    # TODO(Jiayi): Please remove this helper function in the future.
    # Currently, it's only used for testing.
    @staticmethod
    def _Create_memory_allocator(
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        numa_mapping: Optional[NUMAMapping] = None,
    ) -> MemoryAllocatorInterface:
        # NOTE: should remove this function after fixing the unit tests:
        # raise RuntimeError("_Create_memory_allocator is deprecated!")
        extra_config = config.extra_config
        enable_nixl_storage = extra_config is not None and extra_config.get(
            "enable_nixl_storage"
        )

        if enable_nixl_storage:
            # TODO(Jiayi): weird to import from transfer utils.
            # First Party
            from lmcache.v1.transfer_channel.transfer_utils import (
                get_correct_device,
            )

            corrected_device = get_correct_device(
                config.nixl_buffer_device,
                metadata.worker_id,
            )

            buffer = torch.empty(
                config.nixl_buffer_size,
                dtype=torch.uint8,
                device=corrected_device,
            )

            if corrected_device == "cpu":
                torch.cuda.cudart().cudaHostRegister(
                    buffer.data_ptr(), config.nixl_buffer_size, 0
                )
            else:
                logger.info(f"Setting cuda device to {corrected_device} ")
                torch.cuda.set_device(corrected_device)

            return PagedTensorMemoryAllocator(
                buffer,
                [torch.Size(metadata.kv_shape)],
                [metadata.kv_dtype],
                MemoryFormat.KV_2LTD,
            )

        if config.gds_path is not None:
            assert config.cufile_buffer_size is not None
            return CuFileMemoryAllocator(config.cufile_buffer_size * 1024**2)

        max_local_cpu_size = config.max_local_cpu_size
        # save_only_first_rank only works when use mla
        save_only_first_rank = (
            config.get_extra_config_value("save_only_first_rank", metadata.use_mla)
            and metadata.use_mla
        )
        if save_only_first_rank and metadata.is_first_rank():
            # Only the first rank will save the cache,
            # so we need to set it lager than other ranks
            first_rank_max_local_cpu_size = (
                config.extra_config.get(
                    "first_rank_max_local_cpu_size", max_local_cpu_size
                )
                if config.extra_config
                else max_local_cpu_size
            )
            return MixedMemoryAllocator(
                int(first_rank_max_local_cpu_size * 1024**3),
                numa_mapping=numa_mapping,
            )
        return MixedMemoryAllocator(
            int(max_local_cpu_size * 1024**3),
            numa_mapping=numa_mapping,
        )

    @staticmethod
    def _Create_token_database(
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
    ) -> TokenDatabase:
        if config.enable_blending:
            return SegmentTokenDatabase(config, metadata)
        return ChunkedTokenDatabase(config, metadata)

    @classmethod
    def get_or_create(
        cls,
        instance_id: str,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        gpu_connector: Optional[GPUConnectorInterface],
        broadcast_fn: Callable[[torch.Tensor, int], None],
        broadcast_object_fn: Callable[[Any, int], Any],
    ) -> LMCacheEngine:
        """
        Builds a new LMCacheEngine instance if it doesn't already exist for the
        given ID.

        raises: ValueError if the instance already exists with a different
            configuration.
        """
        logger.info(f"Creating LMCacheEngine instance {instance_id}")
        if instance_id not in cls._instances:
            numa_mapping = NUMADetector.get_numa_mapping(config)
            logger.info(f"NUMA mapping for instance {instance_id}: {numa_mapping}")
            token_database = cls._Create_token_database(config, metadata)
            stat_logger = LMCacheStatsLogger(
                metadata,
                log_interval=10,
                config=config,
            )

            engine = LMCacheEngine(
                config,
                metadata,
                token_database,
                gpu_connector,
                broadcast_fn,
                broadcast_object_fn,
            )

            cls._instances[instance_id] = engine
            cls._cfgs[instance_id] = config
            cls._metadatas[instance_id] = metadata
            cls._stat_loggers[instance_id] = stat_logger
            return engine
        else:
            if (
                cls._cfgs[instance_id] != config
                or cls._metadatas[instance_id] != metadata
            ):
                raise ValueError(
                    f"Instance {instance_id} already exists with a different "
                    f"configuration or metadata."
                )
            return cls._instances[instance_id]

    @classmethod
    def get(cls, instance_id: str) -> Optional[LMCacheEngine]:
        """Returns the LMCacheEngine instance associated with the instance ID,
        or None if not found."""
        return cls._instances.get(instance_id)

    @classmethod
    def destroy(cls, instance_id: str) -> None:
        """Close and delete the LMCacheEngine instance by the instance ID"""
        # TODO: unit test for this
        logger.info(f"Destroying LMCacheEngine instance: {instance_id}")

        if instance_id in cls._instances:
            stat_logger = cls._stat_loggers[instance_id]
            try:
                logger.info("Shutting down stats logger...")
                stat_logger.shutdown()
                logger.info("Stats logger shut down successfully")
            except Exception as e:
                logger.error(f"Error shutting down stats logger: {e}")

            engine = cls._instances[instance_id]
            try:
                logger.info("Closing cache engine...")
                engine.close()
                logger.info("Cache engine closed successfully")
            except Exception as e:
                logger.error(f"Error closing cache engine: {e}")

            try:
                logger.info("Cleaning up instance dictionaries...")
                cls._instances.pop(instance_id, None)
                cls._cfgs.pop(instance_id, None)
                cls._metadatas.pop(instance_id, None)
                cls._stat_loggers.pop(instance_id, None)
                logger.info("Instance dictionaries cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up instances: {e}")

            try:
                logger.info("Destroying stats monitor...")
                LMCStatsMonitor.DestroyInstance()
                logger.info("Stats monitor destroyed successfully")
            except Exception as e:
                logger.error(f"Error destroying stats monitor: {e}")

            logger.info(f"LMCacheEngine instance {instance_id} destroyed")
        else:
            logger.warning(f"Instance {instance_id} not found for destruction")
