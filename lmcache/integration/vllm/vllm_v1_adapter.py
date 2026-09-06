# SPDX-License-Identifier: Apache-2.0
# Standard
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generator, Optional, Union
import builtins
import os
import threading
import time

import numpy as np

# Third Party
from vllm.config import (
    VllmConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorWorkerMetadata,
)
from vllm.distributed.parallel_state import (
    get_pp_group,
)
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus
from vllm.v1.utils import record_function_or_nullcontext
from vllm.version import __version__ as VLLM_VERSION
import torch

# First Party
# Use LMCache's own math utilities instead of vllm's
# (avoids dependency on vllm internal changes like https://github.com/vllm-project/vllm/pull/27188)
from lmcache import utils
from lmcache.integration.vllm.utils import (
    ENGINE_NAME,
    apply_mm_hashes_to_token_ids,
    extract_mm_features,
    lmcache_get_or_create_config,
)
from lmcache.integration.vllm.vllm_service_factory import VllmServiceFactory
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor, PrometheusLogger
from lmcache.utils import CacheStoreEvent, _lmcache_nvtx_annotate, cdiv
from lmcache.v1.cache_engine import (
    LMCacheEngine,
    SpecAgentProgressiveLayerwiseBatch,
)
from lmcache.v1.compute.blend import LMCBlenderBuilder
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.config_base import validate_and_set_config_value
from lmcache.v1.manager import LMCacheManager

if TYPE_CHECKING:
    # Third Party
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.multimodal.inputs import PlaceholderRange
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.request import Request

    # First Party
    from lmcache.v1.lookup_client.abstract_client import LookupClientInterface

logger = init_logger(__name__)


def _specagent_record_cuda_event(
    stream: torch.cuda.Stream,
) -> torch.cuda.Event:
    """Create/record events under one process-wide cross-thread lock.

    PyTorch event recording from the model and LMCache producer threads at
    exactly the same time has been observed to deadlock after both enter
    cudaStreamIsCapturing.  The CUDA work remains asynchronous; only the tiny
    host-side event-record calls are serialized.
    """
    lock = getattr(builtins, "_specagent_cuda_event_record_lock", None)
    if lock is None:
        lock = threading.Lock()
        builtins._specagent_cuda_event_record_lock = lock
    with lock:
        event = torch.cuda.Event()
        event.record(stream)
    return event


def _specagent_env_true(env_name: str) -> bool:
    value = os.environ.get(env_name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _specagent_timing_enabled() -> bool:
    return _specagent_env_true("SPECAGENT_LMCACHE_TIMING")


def _specagent_layer_index(layer_name: str) -> Optional[int]:
    for part in layer_name.split("."):
        if part.isdigit():
            return int(part)
    return None


def _specagent_env_layer_enabled(
    env_name: str,
    legacy_env_name: str,
    layer_idx: Optional[int],
) -> bool:
    if layer_idx is None or not os.environ.get("SPECAGENT_KV_FINGERPRINT"):
        return False
    layer_spec = os.environ.get(env_name)
    if layer_spec is None:
        layer_spec = os.environ.get(legacy_env_name, "0")
    layer_spec = layer_spec.strip()
    if layer_spec.lower() in {"all", "*"}:
        return True
    try:
        return int(layer_spec) == layer_idx
    except ValueError:
        pass
    for part in layer_spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            try:
                if int(start_s) <= layer_idx <= int(end_s):
                    return True
            except ValueError:
                continue
        else:
            try:
                if int(part) == layer_idx:
                    return True
            except ValueError:
                continue
    return False


def _specagent_fp_enabled(layer_idx: Optional[int]) -> bool:
    return _specagent_env_layer_enabled(
        "SPECAGENT_KV_FINGERPRINT_LAYERS",
        "SPECAGENT_KV_FINGERPRINT_LAYER",
        layer_idx,
    )


def _specagent_kv_fp_store() -> dict:
    store = getattr(builtins, "_specagent_kv_fingerprints", None)
    if store is None:
        store = {}
        setattr(builtins, "_specagent_kv_fingerprints", store)
    return store


def _specagent_cpu_saved_block_store() -> set[tuple[int, int]]:
    store = getattr(builtins, "_specagent_cpu_saved_blocks", None)
    if store is None:
        store = set()
        setattr(builtins, "_specagent_cpu_saved_blocks", store)
    return store


def _specagent_cpu_saved_chunk_store() -> set[tuple[int, int]]:
    store = getattr(builtins, "_specagent_cpu_saved_chunks", None)
    if store is None:
        store = set()
        setattr(builtins, "_specagent_cpu_saved_chunks", store)
    return store


def _specagent_kv_fingerprint(
    key_tokens: torch.Tensor,
    value_tokens: torch.Tensor,
) -> tuple[float, float, float, float]:
    key_f = key_tokens.float()
    value_f = value_tokens.float()
    return (
        float(key_f.sum().item()),
        float(value_f.sum().item()),
        float((key_f * key_f).sum().item()),
        float((value_f * value_f).sum().item()),
    )


def _specagent_kv_fp_delta(
    lhs: tuple,
    rhs: tuple,
) -> tuple[float, float]:
    sum_delta = max(
        abs(float(lhs[1]) - float(rhs[1])),
        abs(float(lhs[2]) - float(rhs[2])),
    )
    norm_delta = max(
        abs(float(lhs[3]) - float(rhs[3])),
        abs(float(lhs[4]) - float(rhs[4])),
    )
    return sum_delta, norm_delta


@dataclass
class LoadSpec:
    # Number of tokens cached in vLLM
    vllm_cached_tokens: int
    # Number of tokens that are cached in LMCache
    lmcache_cached_tokens: int
    # Whether the scheduler allow us to load the tokens
    can_load: bool


@dataclass
class SaveSpec:
    # Skip already saved tokens
    skip_leading_tokens: int
    # Whether the scheduler allow us to save the tokens
    can_save: bool


@dataclass
class DisaggSpec:
    req_id: str
    receiver_id: str
    receiver_host: str
    receiver_init_port: int
    receiver_alloc_port: int
    is_last_prefill: bool = False
    num_transferred_tokens: int = 0


tmp_disagg_tracker: dict[str, DisaggSpec] = {}


def extract_request_configs(sampling_params: SamplingParams) -> Optional[dict]:
    request_configs = None
    if sampling_params and sampling_params.extra_args is not None:
        if kv_transfer_params := sampling_params.extra_args.get("kv_transfer_params"):
            for k, v in kv_transfer_params.items():
                if k.startswith("lmcache."):
                    if request_configs is None:
                        request_configs = {}
                    request_configs[k] = v
    return request_configs


@dataclass
class RequestTracker:
    # Request id
    req_id: str

    # Total prompt token length
    prompt_len: int

    # The token ids that has been scheduled so far
    token_ids: list[int]

    # The block ids that has been allocated so far
    # NOTE: allocated blocks could be more than the number of tokens
    allocated_block_ids: list[int]

    # The number of tokens that has been saved
    num_saved_tokens: int = 0

    # Disagg spec for the request
    disagg_spec: Optional[DisaggSpec] = None

    # Multimodal hashes and positions
    mm_hashes: Optional[list[str]] = None
    mm_positions: Optional[list["PlaceholderRange"]] = None

    # The configs of the request, includes tags and other configs
    request_configs: Optional[dict] = None

    # Whether the request is in decode phase
    is_decode_phase = False

    # Whether the request cache should be saved
    skip_save: bool = False

    # The number of tokens that are cached in LMCache for this request
    num_lmcache_cached_tokens: int = 0

    @_lmcache_nvtx_annotate
    @staticmethod
    def from_new_request(
        lmcache_config: LMCacheEngineConfig,
        new_request: "NewRequestData",
        num_tokens_to_compute: int,
        lmcache_cached_tokens: int,
        skip_save: bool,
    ) -> "RequestTracker":
        """Create the request tracker from a new request.

        Args:
            lmcache_config (LMCacheEngineConfig): the LMCache engine config.
            new_request (NewRequestData): the new request data.
            num_tokens_to_compute (int): the number of tokens that will
                be 'computed', including the `num_computed_tokens` (vLLM's
                local cache hit) and new tokens that will be scheduled.
            lmcache_cached_tokens (int): the number of tokens that are
                cached in LMCache.
            request_priority (int): the priority of the request
            skip_save (bool): whether the request cache should be saved
        """
        # vLLM 0.9.0 update: request.block_ids changed from list[int] to
        # tuple[list[int]]
        # Need to check the type of request.block_ids

        unfolded_block_ids = []

        if not isinstance(new_request.block_ids[0], list):
            unfolded_block_ids = new_request.block_ids.copy()
        else:
            # According to the vLLM code
            # (https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/
            # sched/scheduler.py#L943),
            # only one KVCacheGroup is supported in connector for now.

            # TODO: Please support multiple KVCacheGroup in connector.
            # NOTE: Also, `update` method in RequestTracker should be
            # updated accordingly.
            unfolded_block_ids = new_request.block_ids[0].copy()

        # NOTE: Initialized in `update_state_after_alloc`
        disagg_spec = tmp_disagg_tracker.pop(new_request.req_id, None)

        request_configs = extract_request_configs(new_request.sampling_params)

        mm_hashes, mm_positions = extract_mm_features(new_request, modify=True)

        return RequestTracker(
            req_id=new_request.req_id,
            prompt_len=len(new_request.prompt_token_ids),
            token_ids=new_request.prompt_token_ids[:num_tokens_to_compute].copy(),
            allocated_block_ids=unfolded_block_ids,
            num_saved_tokens=lmcache_cached_tokens,
            disagg_spec=disagg_spec,
            mm_hashes=mm_hashes,
            mm_positions=mm_positions,
            skip_save=skip_save,
            request_configs=request_configs,
            num_lmcache_cached_tokens=lmcache_cached_tokens,
        )

    def update(
        self,
        new_token_ids: list[int],
        new_block_ids: Union[Optional[tuple[list[int], ...]], list[int]],
        preempted: bool = False,
        lmcache_cached_tokens: int = 0,
        vllm_cached_tokens: int = 0,
        all_token_ids: Optional[list[int]] = None,
    ) -> None:
        """Update the request tracker when a running request is
        scheduled again

        vllm_cached_tokens: the number of tokens that are cached in vLLM
        is only used for preempted requests
        all_token_ids: the full token list from the vLLM request, used to
        restore token_ids for preempted requests to ensure chunk keys match
        """

        if new_block_ids is None:
            # https://github.com/vllm-project/vllm/commit/
            # b029de9902aa3ac58806c8c17776c7074175b6db#
            # diff-cafd89ce8a698a56acb24ada62831cbc7a980782f78a52d1742ba238031f296cL94
            new_block_ids = []
        elif len(new_block_ids) == 0:
            new_block_ids = []
        elif isinstance(new_block_ids, tuple):
            new_block_ids = new_block_ids[0]
        elif isinstance(new_block_ids, list):
            # If input is a list, flatten it to handle potential nesting.
            # This also correctly processes already-flat lists.
            new_block_ids = [
                i
                for elem in new_block_ids
                for i in (elem if isinstance(elem, list) else [elem])
            ]
        else:
            raise ValueError(f"Unsupported new_block_ids type {type(new_block_ids)}")

        if preempted:
            assert all_token_ids is not None, (
                f"Preempted request {self.req_id} has no all_token_ids"
            )
            # the block ids will change after preemption
            self.allocated_block_ids = new_block_ids
            # reset the number of saved tokens
            self.num_saved_tokens = lmcache_cached_tokens
            num_computed_tokens = max(lmcache_cached_tokens, vllm_cached_tokens)

            # FIX: For preempted requests, restore token_ids from the full
            # token list to ensure chunk keys match what was used during
            # lookup. The lookup uses request.all_token_ids, so we need the
            # same tokens for retrieve.
            num_tokens_needed = max(
                num_computed_tokens + len(new_token_ids),
                lmcache_cached_tokens,
            )
            self.token_ids = all_token_ids[:num_tokens_needed]
        else:
            self.allocated_block_ids.extend(new_block_ids)
            self.token_ids.extend(new_token_ids)

        # When a request is scheduled again, and the number of new tokens
        # is 1 (excluding chunked prefill), the request is in decode phase.
        # TODO: Need to further exclude the case of chunked prefill with 1 token.
        if len(new_token_ids) == 1:
            self.is_decode_phase = True


@dataclass
class ReqMeta:
    # Request id
    req_id: str
    # Request tokens
    token_ids: list[int]  # torch.Tensor
    # Slot mapping
    slot_mapping: torch.Tensor

    # Whether is last prefill or not
    is_last_prefill: bool = False

    # Skip save or not
    save_spec: Optional[SaveSpec] = None
    # load_spec
    load_spec: Optional[LoadSpec] = None
    # disagg spec
    disagg_spec: Optional[DisaggSpec] = None
    # the configs of the request
    request_configs: Optional[dict] = None
    # SpecAgent turn-end save emitted after the request actually stopped.
    is_finished_save: bool = False
    # SpecAgent mid-turn preempt save: store the confirmed/verified KV region
    # of a request that is being preempted (but is NOT finished) so it can
    # restore to NORMAL via a CPU hit.  Stored synchronously without a forward,
    # like is_finished_save, but must NOT be reported as finished_sending (the
    # request is still alive; the scheduler frees its blocks itself once the
    # store has landed).
    is_preempt_save: bool = False
    # Identity of this particular mid-turn save. The same live request may be
    # SPEC-preempted more than once.
    specagent_preempt_save_id: int = 0
    # Forced observation-prefill saves can extend a partial LMCache chunk
    # whose old prefix came from CPU scratch and whose new suffix is in paged
    # GPU KV. The store path merges those two sources before publication.
    specagent_tail_merge_prefix_tokens: int = 0
    specagent_observation_tail_save: bool = False

    @staticmethod
    def from_request_tracker(
        tracker: RequestTracker,
        block_size: int,
        lmcache_chunk_size: int = 256,
        load_spec: Optional[LoadSpec] = None,
        discard_partial_chunks: bool = True,
        save_decode_cache: bool = False,
        force_save: bool = False,
    ) -> Optional["ReqMeta"]:
        """Create the request metadata from a request tracker.

        Args:
            tracker (RequestTracker): the request tracker.
            block_size (int): the block size in vLLM.
            lmcache_chunk_size (int): the chunk size for LMCache.
            load_spec (Optional[LoadSpec]): the load spec for KV cache loading.
            discard_partial_chunks (bool): whether to discard partial chunks.
            save_decode_cache (bool): whether to save the cache in decode phase.
            force_save (bool): whether to bypass chunk-boundary/decode save
                skips for a final turn-level flush.

        Returns:
            the request metadata if we need to perform load/save
            operations, None otherwise.
        """
        timing = _specagent_timing_enabled()
        t0 = time.perf_counter() if timing else 0.0
        _ft_t0 = time.perf_counter()
        input_token_ids = tracker.token_ids
        input_token_len = len(input_token_ids)

        is_last_prefill = False
        if input_token_len >= tracker.prompt_len:
            is_last_prefill = True

        # For save operation: do not save if the following condition is met
        # 1. has already been saved before (num_saved_tokens > 0)
        # 2. number of unsaved tokens is not reached the chunk boundary
        # 3. if save_decode_cache is False and it is in decode phase

        skip_leading_tokens = tracker.num_saved_tokens
        chunk_boundary = (
            cdiv(tracker.num_saved_tokens + 1, lmcache_chunk_size) * lmcache_chunk_size
        )

        # NOTE(vladnosiv): for disagg, you cannot skip saving, as saving is a transfer
        # Check if request_configs has lmcache.skip_save set to True
        request_skip = (tracker.request_configs or {}).get("lmcache.skip_save", False)

        explicit_skip = tracker.skip_save or request_skip
        boundary_or_decode_skip = (
            (tracker.num_saved_tokens > 0 and input_token_len < chunk_boundary)
            or (tracker.is_decode_phase and not save_decode_cache)
        )
        skip_save = tracker.disagg_spec is None and (
            explicit_skip or ((not force_save) and boundary_or_decode_skip)
        )

        if skip_save and load_spec is None:
            builtins._lmc_ms_from_tracker = getattr(
                builtins, "_lmc_ms_from_tracker", 0.0) + (
                time.perf_counter() - _ft_t0) * 1000.0
            builtins._lmc_n_from_tracker = getattr(
                builtins, "_lmc_n_from_tracker", 0) + 1
            if timing:
                logger.info(
                    "[SpecAgent][LMCACHE-TIMING] from_tracker req=%s "
                    "result=none reason=skip_save input_tokens=%d "
                    "saved_tokens=%d is_decode=%s elapsed_ms=%.3f",
                    tracker.req_id,
                    input_token_len,
                    tracker.num_saved_tokens,
                    tracker.is_decode_phase,
                    (time.perf_counter() - t0) * 1000.0,
                )
            return None

        # Calculate number of tokens to save based on discard_partial_chunks
        # setting

        # NOTE(vladnosiv): for the input_token_len chunk prefill,
        # we are required to discard partial chunks,
        # as new tokens will be added in the next iteration.
        if not is_last_prefill or discard_partial_chunks:
            num_tokens_to_save = (
                input_token_len // lmcache_chunk_size * lmcache_chunk_size
            )
        else:
            num_tokens_to_save = input_token_len

        if (
            tracker.is_decode_phase
            and tracker.disagg_spec is None
            and os.environ.get("SPECAGENT_ENABLE_DECODE_SAVE_CLAMP")
        ):
            # Decode-time token_ids can include generated suffix tokens before
            # that suffix is a stable prefix for the next turn. Save only the
            # stable prompt here; generated chunks are saved when they re-enter
            # through the next prompt/prefill path.
            stable_prompt_tokens = tracker.prompt_len
            if discard_partial_chunks:
                stable_prompt_tokens = (
                    stable_prompt_tokens // lmcache_chunk_size
                    * lmcache_chunk_size
                )
            num_tokens_to_save = min(num_tokens_to_save, stable_prompt_tokens)

        num_blocks = len(tracker.allocated_block_ids)
        max_slot_tokens = num_blocks * block_size
        if num_tokens_to_save > max_slot_tokens:
            logger.warning(
                "[SpecAgent][SAVE-CLAMP] req=%s save_tokens=%d "
                "slot_capacity=%d blocks=%d block_size=%d reason=no_paged_slot",
                tracker.req_id,
                num_tokens_to_save,
                max_slot_tokens,
                num_blocks,
                block_size,
            )
            num_tokens_to_save = max_slot_tokens
            if discard_partial_chunks:
                num_tokens_to_save = (
                    num_tokens_to_save // lmcache_chunk_size * lmcache_chunk_size
                )
            if skip_leading_tokens > num_tokens_to_save:
                skip_leading_tokens = num_tokens_to_save

        # If we need to save, update the number of saved tokens.
        if not skip_save:
            tracker.num_saved_tokens = num_tokens_to_save
        _sbd = os.environ.get("LMCACHE_SAVEBND_DEBUG")
        if _sbd:
            try:
                import builtins as _b
                _fh = getattr(_b, "_lmc_savebnd_fh", None)
                if _fh is None:
                    _fh = _b._lmc_savebnd_fh = open(_sbd, "a", buffering=1)
                    _b._lmc_savebnd_n = 0
                if _b._lmc_savebnd_n < 8000:
                    _b._lmc_savebnd_n += 1
                    _fh.write(
                        "SAVEBND req=%s input_len=%d prompt_len=%d prev_saved=%d "
                        "to_save=%d max_slot=%d nblocks=%d last_prefill=%s "
                        "decode=%s skip_save=%s clamped=%s\n" % (
                            tracker.req_id, input_token_len, tracker.prompt_len,
                            skip_leading_tokens, num_tokens_to_save, max_slot_tokens,
                            num_blocks, is_last_prefill, tracker.is_decode_phase,
                            skip_save, num_tokens_to_save >= max_slot_tokens,
                        )
                    )
            except Exception:
                pass
        save_spec = SaveSpec(skip_leading_tokens, not skip_save)

        # Calculate the token ids and slot mappings for load and save
        token_ids = input_token_ids[:num_tokens_to_save]

        # If the request has multimodal hashes, apply them to the token ids
        if tracker.mm_hashes:
            # TODO: Optimize this
            token_ids = torch.tensor(token_ids)
            assert tracker.mm_positions is not None, (
                "tracker got mm_hashes but no mm_positions"
            )
            apply_mm_hashes_to_token_ids(
                token_ids, tracker.mm_hashes, tracker.mm_positions
            )
            token_ids = token_ids.tolist()

        if len(token_ids) > num_blocks * block_size:
            logger.error(
                "The number of tokens is more than the number of blocks"
                " for request %s. "
                "Something might be wrong in scheduling logic!",
                tracker.req_id,
            )
            logger.error(
                "Num tokens: %d, num blocks: %d, block size: %d",
                len(token_ids),
                num_blocks,
                block_size,
            )

        # Build the slot mapping with numpy: the equivalent torch broadcast
        # pays multi-ms fork/join overhead in torch's intra-op thread pool
        # (128 threads on this host) for a ~48k-element op, and this runs on
        # the scheduler thread for every fetch request every step.
        block_ids_np = np.asarray(tracker.allocated_block_ids, dtype=np.int64)
        slot_mapping_np = (
            np.arange(block_size, dtype=np.int64).reshape((1, block_size))
            + block_ids_np.reshape((num_blocks, 1)) * block_size
        ).reshape(-1)[: len(token_ids)]
        slot_mapping = torch.from_numpy(slot_mapping_np)
        assert slot_mapping.dtype == torch.long  # TODO: this could be removed

        # For load operation: log if the request is scheduled to load
        if load_spec is not None and load_spec.can_load:
            logger.debug(
                "Scheduled to load %d tokens (%d cached in vLLM) for request %s",
                load_spec.lmcache_cached_tokens,
                load_spec.vllm_cached_tokens,
                tracker.req_id,
            )

        builtins._lmc_ms_from_tracker = getattr(
            builtins, "_lmc_ms_from_tracker", 0.0) + (
            time.perf_counter() - _ft_t0) * 1000.0
        builtins._lmc_n_from_tracker = getattr(
            builtins, "_lmc_n_from_tracker", 0) + 1
        # Note: We keep load_spec even when can_load=False to pass metrics to worker
        req_meta = ReqMeta(
            req_id=tracker.req_id,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            is_last_prefill=is_last_prefill,
            save_spec=save_spec,
            load_spec=load_spec,
            disagg_spec=tracker.disagg_spec,
            request_configs=tracker.request_configs,
        )
        if timing:
            logger.info(
                "[SpecAgent][LMCACHE-TIMING] from_tracker req=%s "
                "result=meta input_tokens=%d save_tokens=%d skip_leading=%d "
                "can_save=%s load_vllm=%s load_lmcache=%s can_load=%s "
                "is_decode=%s force_save=%s elapsed_ms=%.3f",
                tracker.req_id,
                input_token_len,
                len(token_ids),
                save_spec.skip_leading_tokens if save_spec is not None else None,
                save_spec.can_save if save_spec is not None else None,
                load_spec.vllm_cached_tokens if load_spec is not None else None,
                load_spec.lmcache_cached_tokens if load_spec is not None else None,
                load_spec.can_load if load_spec is not None else None,
                tracker.is_decode_phase,
                force_save,
                (time.perf_counter() - t0) * 1000.0,
            )
        return req_meta


@dataclass
class LMCacheConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta] = field(default_factory=list)

    @_lmcache_nvtx_annotate
    def add_request(self, req_meta: ReqMeta) -> None:
        """Add a request to the metadata.

        Args:
            req_meta (ReqMeta): the request metadata.
        """
        self.requests.append(req_meta)


@dataclass
class LMCacheSpecAgentWorkerMetadata(KVConnectorWorkerMetadata):
    """Worker-to-scheduler acknowledgements for exact mid-turn saves."""

    # save key -> (minimum proven prefix tokens, reporting worker count)
    completed_preempt_saves: dict[
        tuple[str, int], tuple[int, int]
    ] = field(
        default_factory=dict
    )

    def aggregate(
        self, other: "KVConnectorWorkerMetadata"
    ) -> "LMCacheSpecAgentWorkerMetadata":
        if not isinstance(other, LMCacheSpecAgentWorkerMetadata):
            return self
        for save_key, (proven, count) in (
            other.completed_preempt_saves.items()
        ):
            previous = self.completed_preempt_saves.get(save_key)
            if previous is None:
                self.completed_preempt_saves[save_key] = (proven, count)
            else:
                self.completed_preempt_saves[save_key] = (
                    min(previous[0], proven),
                    previous[1] + count,
                )
        return self


class LMCacheConnectorV1Impl:
    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        parent: KVConnectorBase_V1,
    ):
        self._parent = parent
        self._vllm_config = vllm_config
        self._role = role
        self.device = vllm_config.device_config.device
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.worker_count = vllm_config.parallel_config.tensor_parallel_size

        # Load and configure LMCache config
        config = lmcache_get_or_create_config()
        assert isinstance(config, LMCacheEngineConfig), (
            "LMCache v1 configuration is should be passed for vLLM v1."
        )
        self._apply_extra_config(config, vllm_config)
        self.config = config

        service_factory = VllmServiceFactory(config, vllm_config, role.name.lower())
        self._manager = LMCacheManager(config, service_factory, connector=self)

        # Start services managed by LMCacheManager
        self._manager.start_services()

        # Initialize connector-specific state
        self._init_connector_state(role, vllm_config, config)

        # Setup metrics for monitoring data structures
        self._setup_metrics()

        logger.info(
            "LMCache initialized for role %s with version %s, "
            "vllm version %s, lmcache cache_engine metadata: %s",
            role,
            utils.get_version(),
            VLLM_VERSION,
            getattr(self.lmcache_engine, "metadata", None),
        )

    def _apply_extra_config(
        self, config: LMCacheEngineConfig, vllm_config: "VllmConfig"
    ) -> None:
        """Apply extra config from vLLM to LMCache config."""
        kv_connector_extra_config = (
            vllm_config.kv_transfer_config.kv_connector_extra_config
        )
        if kv_connector_extra_config:
            for key, value in kv_connector_extra_config.items():
                if key.startswith("lmcache."):
                    config_key = key[8:]  # Remove "lmcache." prefix
                    if validate_and_set_config_value(config, config_key, value):
                        logger.info(
                            "Updated config %s from vLLM extra config",
                            config_key,
                        )

    def _init_connector_state(
        self,
        role: KVConnectorRole,
        vllm_config: "VllmConfig",
        config: LMCacheEngineConfig,
    ) -> None:
        """Initialize connector-specific state variables."""
        self.async_loading = config.enable_async_loading
        self.layerwise_retrievers: list[
            Generator[Optional[torch.Tensor], None, None]
        ] = []
        self._specagent_layerwise_meta: list[dict[str, object]] = []
        self._specagent_layerwise_check_period = 0
        self._specagent_layer_group_size = 1
        self._specagent_waited_layer_groups: set[int] = set()
        self._specagent_prefetch_lock = threading.Lock()
        self._specagent_prefetch_cond = threading.Condition(
            self._specagent_prefetch_lock
        )
        self._specagent_prefetch_future: Future | None = None
        self._specagent_prefetch_target = -1
        self._specagent_prefetch_scan_layer = 0
        self._specagent_prefetch_executor: ThreadPoolExecutor | None = (
            None if role == KVConnectorRole.SCHEDULER
            else ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="specagent-kv-prefetch",
            )
        )
        # CPU-only membership lookup and MemoryObj resolution.  Keep this
        # separate from the layer producer: that producer may run CUDA work,
        # while this executor must remain CUDA-free and may be awaited by it.
        self._specagent_cpu_prepare_executor: ThreadPoolExecutor | None = (
            None if role == KVConnectorRole.SCHEDULER
            else ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="specagent-cpu-prepare",
            )
        )
        self._specagent_async_save_this_step = False
        self._specagent_save_executor: ThreadPoolExecutor | None = (
            None if role == KVConnectorRole.SCHEDULER
            else ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="specagent-kv-save",
            )
        )
        self._specagent_save_futures: list[Future] = []
        self._layerwise_save_storers: dict[
            str, Generator[Optional[torch.Tensor], None, None]
        ] = {}
        self._stats_monitor = LMCStatsMonitor.GetOrCreate()

        # Role-specific initialization
        if role == KVConnectorRole.SCHEDULER:
            self._unfinished_requests: dict[str, "Request"] = {}
            self._specagent_finished_save_metas: dict[str, ReqMeta] = {}
            # Mid-turn preempt saves registered by the scheduler; emitted into
            # the connector metadata (like finished saves) and stored on the
            # worker WITHOUT a finished_sending report.
            self._specagent_preempt_save_metas: dict[str, ReqMeta] = {}
            self._specagent_preempt_save_acks: dict[
                tuple[str, int], int
            ] = {}
            # A worker-proven, CPU-pinned prefix committed immediately before
            # a running SPEC request is preempted.  The scheduler lookup
            # service can still hold the answer from before that synchronous
            # store; use this exact coverage until the restore allocation
            # consumes it.
            self._specagent_preempt_forced_lookup: dict[str, int] = {}
        else:
            self.use_layerwise = config.use_layerwise
            self.enable_blending = config.enable_blending
            self._specagent_finished_save_completed: set[str] = set()
            self._specagent_preempt_save_completed: dict[
                tuple[str, int], tuple[int, int]
            ] = {}

            if self.enable_blending:
                assert self.lmcache_engine is not None
                assert self.lmcache_engine.gpu_connector is not None, (
                    "GPU connector must be available for blending"
                )
                self.blender = LMCBlenderBuilder.get_or_create(
                    ENGINE_NAME,
                    self.lmcache_engine,
                    self.lmcache_engine.gpu_connector,
                    config,
                )

        # Legacy compatibility check
        self._check_legacy_register_kv_caches()

        self.kv_caches: dict[str, torch.Tensor] = {}
        self._block_size = vllm_config.cache_config.block_size
        self.load_specs: dict[str, LoadSpec] = {}
        self.kv_cache_manager: Optional["KVCacheManager"] = None
        self._request_trackers: dict[str, RequestTracker] = {}

        self._discard_partial_chunks = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "discard_partial_chunks", False
            )
            or not config.save_unfull_chunk
        )

        self._lmcache_chunk_size = config.chunk_size

        self.skip_last_n_tokens = vllm_config.kv_transfer_config.get_from_extra_config(
            "skip_last_n_tokens", 0
        )

        self.num_layers = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.current_layer = 0

        self.force_skip_save = bool(os.environ.get("LMCACHE_FORCE_SKIP_SAVE", False))
        self._requests_priority: dict[str, int] = {}
        self._invalid_block_ids: set[int] = set()

    def _check_legacy_register_kv_caches(self) -> None:
        """Check for legacy connector without register_kv_caches implementation."""
        if self.lmcache_engine is None:
            return

        child_class = self._parent.__class__
        parent_class = KVConnectorBase_V1
        child_method = getattr(child_class, "register_kv_caches", None)
        parent_method = getattr(parent_class, "register_kv_caches", None)

        if child_method is None or parent_method is None:
            implements = False
        else:
            implements = child_method is not parent_method

        if not implements:
            logger.warning(
                "Please use the latest lmcache connector, otherwise some "
                "features may not work, such as DSA"
            )
            self._manager.post_init()

    # ==================== Property Accessors ====================

    @property
    def lmcache_engine(self) -> Optional[LMCacheEngine]:
        """Get the LMCache engine instance from manager."""
        return self._manager.lmcache_engine

    @property
    def lmcache_engine_metadata(self):
        """Get the LMCache engine metadata from manager."""
        return self._manager.lmcache_engine_metadata

    @property
    def lookup_client(self) -> Optional["LookupClientInterface"]:
        """Get the lookup client from manager."""
        return self._manager.lookup_client

    @property
    def lookup_server(self):
        """Get the lookup server from manager."""
        return self._manager.lookup_server

    def _setup_metrics(self):
        """Setup metrics for monitoring data structures in the connector."""
        prometheus_logger = PrometheusLogger.GetInstanceOrNone()
        if prometheus_logger is None:
            logger.warning(
                "PrometheusLogger is not initialized, "
                "connector metrics will not be collected"
            )
            return

        # Set up metrics for scheduler-specific and general data structures
        metrics_map = {
            "_unfinished_requests": "scheduler_unfinished_requests_count",
            "load_specs": "connector_load_specs_count",
            "_request_trackers": "connector_request_trackers_count",
            "kv_caches": "connector_kv_caches_count",
            "layerwise_retrievers": "connector_layerwise_retrievers_count",
            "_invalid_block_ids": "connector_invalid_block_ids_count",
            "_requests_priority": "connector_requests_priority_count",
        }

        for attr_name, metric_name in metrics_map.items():
            if hasattr(self, attr_name):
                metric = getattr(prometheus_logger, metric_name)
                # Use a default argument in the lambda to capture
                # the current value of `attr_name`
                # to avoid issues with late binding in closures.
                metric.set_function(lambda name=attr_name: len(getattr(self, name)))

    def get_inference_info(self) -> dict:
        """Get inference information including vLLM config and related details.

        Returns:
            dict: Dictionary containing inference information
        """
        # Get vLLM config information
        vllm_config = self._vllm_config

        # Use vLLM config's string representation and add specific configs
        inference_info = {
            "vllm_version": VLLM_VERSION,
            "lmcache_version": utils.get_version(),
            "vllm_config": str(vllm_config),
            "model_config": {
                "model": getattr(vllm_config.model_config, "model", None),
                "dtype": str(getattr(vllm_config.model_config, "dtype", None)),
                "max_model_len": getattr(
                    vllm_config.model_config, "max_model_len", None
                ),
                "vocab_size": getattr(vllm_config.model_config, "vocab_size", None),
                "num_layers": getattr(
                    vllm_config.model_config, "get_num_layers", lambda _: None
                )(vllm_config.parallel_config),
                "num_attention_heads": getattr(
                    vllm_config.model_config, "get_num_attention_heads", lambda _: None
                )(vllm_config.parallel_config),
                "num_kv_heads": getattr(
                    vllm_config.model_config, "get_num_kv_heads", lambda _: None
                )(vllm_config.parallel_config),
                "head_size": getattr(
                    vllm_config.model_config, "get_head_size", lambda: None
                )(),
            },
            "cache_config": {
                "block_size": getattr(vllm_config.cache_config, "block_size", None),
                "cache_dtype": str(
                    getattr(vllm_config.cache_config, "cache_dtype", None)
                ),
                "gpu_memory_utilization": getattr(
                    vllm_config.cache_config, "gpu_memory_utilization", None
                ),
                "swap_space": getattr(vllm_config.cache_config, "swap_space", None),
                "enable_prefix_caching": getattr(
                    vllm_config.cache_config, "enable_prefix_caching", None
                ),
            },
        }

        return inference_info

    def get_inference_version(self) -> str:
        """Get vLLM version information.

        Returns:
            str: vLLM version string
        """
        return VLLM_VERSION

    def _build_kv_layer_groups(self):
        # Build KV layer groups structure if not already built
        if self.lmcache_engine is not None:
            assert len(self.kv_caches) > 0
            kv_layer_groups_manager = (
                self.lmcache_engine.metadata.kv_layer_groups_manager
            )
            kv_layer_groups_manager.build_kv_layer_groups(self.kv_caches)

    # TODO(chunxiaozheng): in the latest lmcache_connector, we use `register_kv_caches`
    #  to init self.kv_caches, we keep it in order to be compatible with old versions
    #  and will be removed in the future.
    @_lmcache_nvtx_annotate
    def _init_kv_caches_from_forward_context(self, forward_context: "ForwardContext"):
        for layer_name in forward_context.no_compile_layers:
            attn_layer = forward_context.no_compile_layers[layer_name]
            if not hasattr(attn_layer, "kv_cache"):
                logger.debug("The layer %s does not have kv_cache, skip it", layer_name)
                continue

            if layer_name not in self.kv_caches:
                self.kv_caches[layer_name] = attn_layer.kv_cache[
                    forward_context.virtual_engine
                ]

        self._build_kv_layer_groups()

    ####################
    # Worker side APIs
    ####################
    @_lmcache_nvtx_annotate
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        logger.info("Registering KV caches")
        # TODO(chunxiaozheng): `_init_kv_caches_from_forward_context` is
        #  not called, we should consider removing it.
        assert len(self.kv_caches) == 0 and len(kv_caches) > 0
        self.kv_caches = kv_caches
        self._build_kv_layer_groups()
        self._manager.post_init()

    @_lmcache_nvtx_annotate
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """Start loading the KV cache from the connector buffer to vLLM's
        paged KV buffer.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.
        """
        detailed_nvtx = bool(
            os.environ.get("SPECAGENT_DETAILED_RETRIEVE_NVTX")
        )
        detail_record = (
            record_function_or_nullcontext if detailed_nvtx else nullcontext
        )
        self.current_layer = 0
        with detail_record("lmcache adapter detail: join_prefetch_future"):
            self._specagent_join_prefetch_future()
        with detail_record("lmcache adapter detail: join_save_futures"):
            self._specagent_join_save_futures()

        gpu_connector = getattr(
            getattr(self, "lmcache_engine", None), "gpu_connector", None
        )
        reap_completed_staging = getattr(
            gpu_connector, "specagent_reap_completed_staging", None
        )
        if callable(reap_completed_staging):
            with detail_record("lmcache adapter detail: reap_staging"):
                reap_completed_staging()

        if len(self.kv_caches) == 0:
            logger.warning(
                "Please update LMCacheConnector, "
                "use register_kv_caches to init kv_caches"
            )
            self._init_kv_caches_from_forward_context(forward_context)

        metadata = self._parent._get_connector_metadata()
        assert isinstance(metadata, LMCacheConnectorMetadata)
        timing = _specagent_timing_enabled()
        t0 = time.perf_counter() if timing else 0.0

        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())
        sa_extra = forward_context.additional_kwargs or {}
        sa_dest_kvcaches = sa_extra.get("destination_kvcaches")
        sa_dest_slot_mapping = sa_extra.get("destination_slot_mapping") or {}
        sa_scratch_reuse_event = sa_extra.get(
            "specagent_scratch_reuse_event"
        )
        sa_fetch_reqs = sa_extra.get("specagent_fetch_reqs") or {}
        sa_fetch_token_ranges = sa_extra.get("specagent_fetch_token_ranges") or {}
        sa_cascade_maps = sa_extra.get("specagent_cascade_maps") or {}
        sa_fetch_buf_blocks = sa_extra.get("specagent_fetch_buf_blocks")
        sa_fetch_paged_blocks = sa_extra.get("specagent_fetch_paged_blocks")
        # Layerwise stores use deferred commit: each layer only enqueues D2H
        # work on the store stream, and the single synchronization/publication
        # happens in wait_for_save.  Running those cheap enqueue steps through
        # a Python executor makes the model and save threads contend once per
        # layer, stretching both compute and fetch spans for short forwards.
        # Keep submission on the model thread; the GPU transfer itself remains
        # asynchronous and overlaps through the dedicated store stream.
        self._specagent_async_save_this_step = False
        sa_cpu_prefetch_ahead_layers = max(
            0,
            int(sa_extra.get("specagent_cpu_prefetch_ahead_layers", 0) or 0),
        )
        sa_full_plan_fetch = bool(sa_extra.get("specagent_full_plan_fetch"))

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            logger.debug("In connector.start_load_kv, but the attn_metadata is None")
            if timing:
                logger.info(
                    "[SpecAgent][LMCACHE-TIMING] start_load_kv "
                    "result=no_attn_metadata metadata_reqs=%d elapsed_ms=%.3f",
                    len(metadata.requests),
                    (time.perf_counter() - t0) * 1000.0,
                )
            return

        assert self.lmcache_engine is not None
        if sa_scratch_reuse_event is not None:
            load_stream = getattr(
                getattr(self.lmcache_engine, "gpu_connector", None),
                "load_stream",
                None,
            )
            if load_stream is not None:
                load_stream.wait_event(sa_scratch_reuse_event)

        self.layerwise_retrievers = []
        self._specagent_layerwise_meta = []
        sa_lead = max(1, int(sa_extra.get("layer_prefetch_granularity", 2)))
        self._specagent_prefetch_lead = sa_lead
        self._specagent_layer_group_size = sa_lead
        self._specagent_waited_layer_groups.clear()
        self._specagent_prefetch_scan_layer = 0
        self._specagent_layerwise_check_period = max(1, 2 * sa_lead)
        sa_verbose = (
            _specagent_env_true("SPECAGENT_VERBOSE")
            or _specagent_env_true("SPECAGENT_FETCH_DEBUG")
            or _specagent_env_true("SPECAGENT_BW_PROFILE")
        )
        if sa_fetch_reqs and sa_verbose:
            logger.info(
                "[SpecAgent][FETCH-ADAPTER] fetch_reqs=%s metadata_reqs=%s "
                "has_dest=%s has_slot=%s",
                list(sa_fetch_reqs.keys()),
                [req.req_id for req in metadata.requests],
                sa_dest_kvcaches is not None,
                list(sa_dest_slot_mapping.keys()),
            )

        for idx, request in enumerate(metadata.requests):
            if request.load_spec is None or not request.load_spec.can_load:
                if (
                    request.req_id in sa_fetch_reqs
                    and os.environ.get("SPECAGENT_FETCH_DEBUG")
                ):
                    logger.warning(
                        "[SpecAgent][FETCH-MASK] skipped req=%s reason=no_load_spec "
                        "fetch_reason=%s",
                        request.req_id,
                        sa_fetch_reqs.get(request.req_id),
                    )
                continue
            last_idx = idx

        for idx, request in enumerate(metadata.requests):
            # Update metrics for all requests that have a load_spec
            if request.load_spec is not None:
                self._stats_monitor.update_interval_vllm_hit_tokens(
                    request.load_spec.vllm_cached_tokens
                )
                self._stats_monitor.update_interval_prompt_tokens(
                    len(request.token_ids)
                )

            if request.load_spec is None or not request.load_spec.can_load:
                if (
                    request.req_id in sa_fetch_reqs
                    and os.environ.get("SPECAGENT_FETCH_DEBUG")
                ):
                    logger.warning(
                        "[SpecAgent][FETCH-MASK] skipped req=%s reason=no_load_spec "
                        "fetch_reason=%s",
                        request.req_id,
                        sa_fetch_reqs.get(request.req_id),
                    )
                continue

            tokens = request.token_ids
            # TODO: have a pre-allocated buffer to hold the slot_mappings
            specagent_fetch_reason = str(sa_fetch_reqs.get(request.req_id, ""))
            use_paged_restore_target = specagent_fetch_reason in {
                "verify_paged_restore",
                "observation_paged_restore",
            }
            use_specagent_buf = (
                sa_dest_kvcaches is not None
                and request.req_id in sa_fetch_reqs
                and request.req_id in sa_dest_slot_mapping
                and not use_paged_restore_target
            )
            use_specagent_slot_mapping = (
                request.req_id in sa_fetch_reqs
                and request.req_id in sa_dest_slot_mapping
            )
            use_specagent_sparse_mask = (
                request.req_id in sa_fetch_reqs
                and request.req_id in sa_fetch_token_ranges
            )
            if use_specagent_buf:
                target_kvcaches = sa_dest_kvcaches
                slot_mapping = sa_dest_slot_mapping[request.req_id].to(self.device)
            else:
                target_kvcaches = kvcaches
                if use_specagent_slot_mapping:
                    slot_mapping = sa_dest_slot_mapping[request.req_id].to(
                        self.device
                    )
                else:
                    slot_mapping = request.slot_mapping.to(self.device)
            if not use_specagent_buf and not use_specagent_slot_mapping:
                assert len(tokens) == len(slot_mapping)

            token_mask = torch.ones(len(tokens), dtype=torch.bool)
            masked_token_count = (
                request.load_spec.vllm_cached_tokens
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )
            token_mask[:masked_token_count] = False

            lmcache_cached_tokens = request.load_spec.lmcache_cached_tokens
            if use_specagent_sparse_mask:
                segment_plan_context = detail_record(
                    "lmcache adapter detail: sparse_segment_plan "
                    f"req={request.req_id} tokens={len(tokens)}"
                )
                segment_plan_context.__enter__()
                clipped_ranges = []
                for start, end in sa_fetch_token_ranges.get(request.req_id, []):
                    start = max(0, int(start))
                    end = min(
                        int(end),
                        len(tokens),
                        len(slot_mapping),
                    )
                    if start < end:
                        clipped_ranges.append((start, end))
                chunk_segments = set()
                spec_map = sa_cascade_maps.get(request.req_id) or {}
                # Fetch only the previous-prefix region described by the
                # SpecAgent block map.  The current observation/prefill suffix
                # is already being computed by vLLM in this step; including it
                # in the LMCache mask makes the final partial chunk request a
                # different key from the CPU-saved prefix.
                prefix_len = int(spec_map.get("prefix_len", 0) or 0)
                fetch_prefix_limit = min(
                    len(tokens), len(slot_mapping), lmcache_cached_tokens
                )
                if prefix_len > 0:
                    fetch_prefix_limit = min(fetch_prefix_limit, prefix_len)
                for start, end in clipped_ranges:
                    chunk_start = (
                        start // self._lmcache_chunk_size
                        * self._lmcache_chunk_size
                    )
                    chunk_end = cdiv(end, self._lmcache_chunk_size) * (
                        self._lmcache_chunk_size
                    )
                    chunk_end = min(
                        chunk_end,
                        len(tokens),
                        len(slot_mapping),
                        fetch_prefix_limit,
                    )
                    if chunk_start < chunk_end:
                        cur = chunk_start
                        while cur < chunk_end:
                            nxt = min(cur + self._lmcache_chunk_size,
                                      chunk_end)
                            chunk_segments.add((cur, nxt))
                            cur = nxt
                effective_chunk_segments = sorted(chunk_segments)
                if effective_chunk_segments:
                    merged_segments = []
                    cur_start, cur_end = effective_chunk_segments[0]
                    for seg_start, seg_end in effective_chunk_segments[1:]:
                        if seg_start <= cur_end:
                            cur_end = max(cur_end, seg_end)
                        else:
                            merged_segments.append((cur_start, cur_end))
                            cur_start, cur_end = seg_start, seg_end
                    merged_segments.append((cur_start, cur_end))
                    effective_chunk_segments = merged_segments
                fetch_tokens = sum(
                    end - start for start, end in effective_chunk_segments
                )
                segment_plan_context.__exit__(None, None, None)
                if sa_verbose:
                    logger.info(
                        "[SpecAgent][FETCH-MASK] req=%s fetch_reason=%s "
                        "ranges=%s chunk_segments=%s fetch_tokens=%d "
                        "lmcache_cached_tokens=%d vllm_cached_tokens=%d "
                        "slot_len=%d target=%s",
                        request.req_id,
                        specagent_fetch_reason,
                        clipped_ranges,
                        effective_chunk_segments,
                        fetch_tokens,
                        lmcache_cached_tokens,
                        request.load_spec.vllm_cached_tokens,
                        len(slot_mapping),
                        "scratch" if use_specagent_buf else "paged",
                    )
                if not effective_chunk_segments:
                    if os.environ.get("SPECAGENT_FETCH_DEBUG"):
                        logger.warning(
                            "[SpecAgent][FETCH-MASK] skipped req=%s "
                            "reason=no_missing_ranges fetch_reason=%s "
                            "lmcache_cached_tokens=%d",
                            request.req_id,
                            sa_fetch_reqs.get(request.req_id),
                            lmcache_cached_tokens,
                        )
                    continue

                shared_chunk_plans = None
                cpu_prepare_future: Any = None
                if self.use_layerwise:
                    with detail_record(
                        "lmcache adapter detail: shared_sparse_chunk_plan "
                        f"req={request.req_id} tokens={len(tokens)} "
                        f"segments={len(effective_chunk_segments)}"
                    ):
                        shared_chunk_plans = (
                            self.lmcache_engine.plan_sparse_layerwise_chunks(
                                tokens,
                                effective_chunk_segments,
                                request.request_configs,
                            )
                        )
                    cpu_prepare_executor = (
                        self._specagent_cpu_prepare_executor
                    )
                    if (
                        sa_full_plan_fetch
                        and shared_chunk_plans
                        and cpu_prepare_executor is not None
                        and not _specagent_env_true(
                            "SPECAGENT_DISABLE_ASYNC_CPU_PREPARE"
                        )
                    ):
                        with detail_record(
                            "lmcache adapter detail: submit_cpu_prepare "
                            f"req={request.req_id} "
                            f"segments={len(shared_chunk_plans)}"
                        ):
                            if _specagent_env_true(
                                "SPECAGENT_PROGRESSIVE_CPU_PREPARE"
                            ):
                                builtins._specagent_progressive_batches = (
                                    getattr(
                                        builtins,
                                        "_specagent_progressive_batches",
                                        0,
                                    )
                                    + 1
                                )
                                progressive_batch = (
                                    SpecAgentProgressiveLayerwiseBatch(
                                        len(shared_chunk_plans)
                                    )
                                )
                                cpu_prepare_executor.submit(
                                    self.lmcache_engine
                                    .prepare_sparse_layerwise_plans_progressive,
                                    shared_chunk_plans,
                                    progressive_batch,
                                )
                                cpu_prepare_future = progressive_batch
                            else:
                                cpu_prepare_future = cpu_prepare_executor.submit(
                                    self.lmcache_engine
                                    .prepare_sparse_layerwise_plans,
                                    shared_chunk_plans,
                                )

                # The storage lookup below is the authoritative proof: it
                # derives LMCache's chained keys and requires every requested
                # layer/chunk to exist. Keep the old Python tuple-prefix
                # ledger scan only as an opt-in diagnostic. It duplicates the
                # lookup and is O(N^2) for long contexts.
                verify_legacy_save_proof = _specagent_env_true(
                    "SPECAGENT_VERIFY_LEGACY_SAVE_PROOF"
                )
                saved_chunk_store = None
                prefix_hash_by_chunk_end: dict[int, int] = {}
                hash_tokens_tuple: tuple[int, ...] = ()
                if verify_legacy_save_proof:
                    saved_chunk_store = _specagent_cpu_saved_chunk_store()
                    with detail_record(
                        "lmcache adapter detail: proof_tokens_tuple "
                        f"req={request.req_id} tokens={len(tokens)}"
                    ):
                        hash_tokens_tuple = tuple(int(t) for t in tokens)

                for seg_i, (seg_start, seg_end) in enumerate(
                    effective_chunk_segments
                ):
                    never_saved_chunks = []
                    if verify_legacy_save_proof:
                        assert saved_chunk_store is not None
                        with detail_record(
                            "lmcache adapter detail: saved_proof_scan "
                            f"req={request.req_id} seg={seg_start}-{seg_end}"
                        ):
                            for chunk_end in range(
                                seg_start + self._lmcache_chunk_size,
                                seg_end + 1,
                                self._lmcache_chunk_size,
                            ):
                                token_hash = prefix_hash_by_chunk_end.get(
                                    chunk_end
                                )
                                if token_hash is None:
                                    token_hash = hash(
                                        hash_tokens_tuple[:chunk_end]
                                    )
                                    prefix_hash_by_chunk_end[chunk_end] = (
                                        token_hash
                                    )
                                if (
                                    (chunk_end, token_hash)
                                    not in saved_chunk_store
                                ):
                                    never_saved_chunks.append(
                                        (
                                            chunk_end
                                            - self._lmcache_chunk_size,
                                            chunk_end,
                                        )
                                    )
                    if verify_legacy_save_proof and never_saved_chunks:
                        raise RuntimeError(
                            "[SpecAgent][FETCH-UNAVAILABLE] req=%s "
                            "fetch_reason=%s segment=[%d,%d) "
                            "never_saved_chunks=%s. Scheduler advertised "
                            "these missing blocks as CPU-fetchable, but "
                            "LMCache save path has no proof that the aligned "
                            "chunks were ever saved in this process."
                            % (
                                request.req_id,
                                specagent_fetch_reason,
                                seg_start,
                                seg_end,
                                never_saved_chunks,
                            )
                        )
                    seg_token_mask = None
                    if not self.use_layerwise:
                        seg_token_mask = torch.ones(
                            seg_end, dtype=torch.bool
                        )
                        seg_token_mask[:seg_start] = False
                    num_expected_tokens = seg_end - seg_start
                    expected_mib = (
                        num_expected_tokens
                        * self._specagent_kv_bytes_per_layer_token()
                        / (1024 * 1024)
                    )
                    specagent_write_token_mask = None
                    if use_paged_restore_target:
                        specagent_write_token_mask = torch.zeros(
                            seg_end, dtype=torch.bool
                        )
                        for range_start, range_end in clipped_ranges:
                            write_start = max(seg_start, range_start)
                            write_end = min(seg_end, range_end)
                            if write_start < write_end:
                                specagent_write_token_mask[
                                    write_start:write_end
                                ] = True
                    if self.use_layerwise:
                        # SpecAgent scratch attention may read every fetched
                        # segment in the current layer.  The generic LMCache
                        # path only synchronizes the final retriever, but that
                        # leaves earlier scratch segments on the load stream
                        # without a current-stream dependency.  Correctness
                        # requires the layer's scratch blocks to be visible
                        # before attention consumes them; later this can be
                        # relaxed with per-segment events/double buffering.
                        sync = True
                        layer_events: list[Any] = [None] * self.num_layers
                        defer_layer_submit = bool(
                            use_specagent_buf
                            and not use_paged_restore_target
                            and not os.environ.get(
                                "SPECAGENT_DISABLE_FUSED_LAYER_SUBMIT"
                            )
                        )
                        if os.environ.get("SPECAGENT_FUSED_TRANSFER_DEBUG"):
                            logger.warning(
                                "[SpecAgent][FUSED-ADMISSION] req=%s "
                                "reason=%s scratch=%s paged_restore=%s "
                                "defer=%s segments=%s",
                                request.req_id,
                                specagent_fetch_reason,
                                use_specagent_buf,
                                use_paged_restore_target,
                                defer_layer_submit,
                                effective_chunk_segments,
                            )
                        with record_function_or_nullcontext(
                            "lmcache adapter: specagent retrieve_layer "
                            f"req={request.req_id} seg={seg_start}-{seg_end} "
                            f"expected_tokens={num_expected_tokens} "
                            f"expected_layer_mib={expected_mib:.1f}"
                        ):
                            layerwise_retriever = self.lmcache_engine.retrieve_layer(
                                tokens[:seg_end],
                                seg_token_mask,
                                kvcaches=target_kvcaches,
                                slot_mapping=slot_mapping[:seg_end],
                                vllm_cached_tokens=seg_start,
                                sync=sync,
                                req_id=request.req_id,
                                specagent_layer_events=layer_events,
                                specagent_defer_layer_submit=defer_layer_submit,
                                specagent_write_token_mask=(
                                    specagent_write_token_mask
                                ),
                                specagent_expected_tokens=num_expected_tokens,
                                specagent_prefix_tokens=seg_start,
                                specagent_chunk_plan=(
                                    shared_chunk_plans[seg_i]
                                    if shared_chunk_plans is not None else None
                                ),
                                specagent_cpu_prefetch_ahead_layers=(
                                    sa_cpu_prefetch_ahead_layers
                                ),
                                specagent_full_plan_fetch=sa_full_plan_fetch,
                                specagent_prepared_plan_future=(
                                    cpu_prepare_future
                                ),
                                specagent_prepared_segment_index=seg_i,
                            )
                        # The first generator step performs the expensive
                        # token/chunk lookup and submits CPU get work.  Let
                        # the async producer execute it so start_load_kv does
                        # not block before the model reaches layer 0.  The
                        # fetched-token validation still runs in
                        # _specagent_advance_retrievers_until before attention
                        # can consume the layer.
                        self.layerwise_retrievers.append(layerwise_retriever)
                        self._specagent_layerwise_meta.append({
                            "req_id": request.req_id,
                            "segment": (seg_start, seg_end),
                            "is_specagent": True,
                            "fetch_reason": specagent_fetch_reason,
                            "is_paged_restore": use_paged_restore_target,
                            "defer_layer_submit": defer_layer_submit,
                            "expected_tokens": num_expected_tokens,
                            "expected_fetch_tokens": num_expected_tokens,
                            "layer_events": layer_events,
                            "copy_to_paged": (
                                bool(
                                    use_specagent_buf
                                    and str(sa_fetch_reqs.get(request.req_id))
                                    in {
                                        "observation_paged_restore",
                                        "verify_paged_restore",
                                    }
                                    and sa_fetch_buf_blocks is not None
                                    and sa_fetch_paged_blocks is not None
                                )
                            ),
                            "src_kvcaches": target_kvcaches,
                            "dst_kvcaches": kvcaches,
                            "src_blocks": sa_fetch_buf_blocks,
                            "dst_blocks": sa_fetch_paged_blocks,
                        })
                    else:
                        assert seg_token_mask is not None
                        sa_retrieve_mask = seg_token_mask
                        if specagent_write_token_mask is not None:
                            # Non-layerwise retrieve() has no separate write
                            # mask, so restrict the fetch/scatter mask itself
                            # to the missing ranges.  Without this the
                            # chunk-aligned segment overwrites resident
                            # tokens with the older CPU copy and, worse,
                            # spills into logical blocks that have no real
                            # destination -- their slot-map filler is
                            # physical block 0, so the spill scribbles over
                            # the shared null/hole block.
                            sa_retrieve_mask = (
                                seg_token_mask & specagent_write_token_mask
                            )
                        ret_token_mask = self.lmcache_engine.retrieve(
                            tokens[:seg_end],
                            sa_retrieve_mask,
                            kvcaches=target_kvcaches,
                            slot_mapping=slot_mapping[:seg_end],
                            vllm_cached_tokens=seg_start,
                            request_configs=request.request_configs,
                            req_id=request.req_id,
                        )
                        num_retrieved_tokens = ret_token_mask.sum().item()
                        num_expected_tokens = int(
                            sa_retrieve_mask.sum().item())
                        if num_retrieved_tokens < num_expected_tokens:
                            raise RuntimeError(
                                "Request %s retrieved %d tokens, expected %d "
                                "for SpecAgent segment [%d, %d)"
                                % (
                                    request.req_id,
                                    num_retrieved_tokens,
                                    num_expected_tokens,
                                    seg_start,
                                    seg_end,
                                )
                )
                continue

            if self.use_layerwise:
                if idx == last_idx:
                    sync = True
                else:
                    sync = False
                # NOTE(Jiayi): Perform blending before layerwise prefix caching
                if self.enable_blending:
                    # TODO(Jiayi): Need to make prefix caching and blending compatible
                    self.blender.blend(
                        tokens[:lmcache_cached_tokens],
                        token_mask[:lmcache_cached_tokens],
                        kvcaches=target_kvcaches,
                        slot_mapping=slot_mapping[:lmcache_cached_tokens],
                        vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
                    )
                else:
                    with record_function_or_nullcontext(
                        f"lmcache adapter: retrieve_layer req={request.req_id}"
                    ):
                        layerwise_retriever = self.lmcache_engine.retrieve_layer(
                            tokens[:lmcache_cached_tokens],
                            token_mask[:lmcache_cached_tokens],
                            kvcaches=target_kvcaches,
                            slot_mapping=slot_mapping[:lmcache_cached_tokens],
                            vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
                            sync=sync,
                        )
                    # NOTE: retrieve for two layers at the first layer
                    with record_function_or_nullcontext(
                        f"lmcache adapter: retrieve prime0 req={request.req_id}"
                    ):
                        next(layerwise_retriever)
                    with record_function_or_nullcontext(
                        f"lmcache adapter: retrieve prime1 req={request.req_id}"
                    ):
                        next(layerwise_retriever)
                    self.layerwise_retrievers.append(layerwise_retriever)
                    self._specagent_layerwise_meta.append(None)
            else:
                ret_token_mask = self.lmcache_engine.retrieve(
                    tokens[:lmcache_cached_tokens],
                    token_mask[:lmcache_cached_tokens],
                    kvcaches=target_kvcaches,
                    slot_mapping=slot_mapping[:lmcache_cached_tokens],
                    vllm_cached_tokens=request.load_spec.vllm_cached_tokens,
                    request_configs=request.request_configs,
                    req_id=request.req_id,
                )

                # Check the result
                num_retrieved_tokens = ret_token_mask.sum().item()
                num_expected_tokens = (
                    lmcache_cached_tokens - request.load_spec.vllm_cached_tokens
                )
                if num_retrieved_tokens < num_expected_tokens:
                    logger.error(
                        "Request %s"
                        "The number of retrieved tokens is less than the "
                        "expected number of tokens! This should not happen!",
                        request.req_id,
                    )
                    logger.error(
                        "Num retrieved tokens: %d, num expected tokens: %d",
                        num_retrieved_tokens,
                        num_expected_tokens,
                    )
                    """
                    Report failed block IDs in case of partial failure.
                    """
                    missing_blocks = self.record_failed_blocks(
                        request.req_id,
                        token_mask[:lmcache_cached_tokens],
                        ret_token_mask,
                        slot_mapping[:lmcache_cached_tokens],
                    )
                    self._invalid_block_ids.update(missing_blocks)

        profile_cascade_only = bool(
            self._specagent_has_layerwise_fetch()
            and _specagent_env_true("SPECAGENT_PROFILE_CASCADE_ONLY")
        )
        if profile_cascade_only:
            # Performance-only ablation: preserve the exact scratch-cascade
            # attention path while removing its layer producer and all transfer
            # dependencies.  Scratch contents are intentionally unspecified;
            # outputs from a run with this knob must never be used for
            # correctness or acceptance measurements.
            ready = _specagent_record_cuda_event(
                torch.cuda.current_stream(self.device)
            )
            for meta in self._specagent_layerwise_meta:
                events = (
                    meta.get("layer_events")
                    if isinstance(meta, dict)
                    else None
                )
                if isinstance(events, list):
                    events[:] = [ready] * len(events)
            self.layerwise_retrievers = []
            self._specagent_prefetch_scan_layer = self.num_layers
        elif self._specagent_all_fetches_paged_restore():
            self._specagent_request_async_prefetch(self.num_layers - 1)
        elif self._specagent_has_layerwise_fetch():
            group_size = max(
                1,
                int(getattr(self, "_specagent_layer_group_size", 1) or 1),
            )
            initial_group_end = min(
                2 * group_size - 1,
                self.num_layers - 1,
            )
            self._specagent_request_async_prefetch(initial_group_end)

        if timing:
            loadable = sum(
                1
                for req in metadata.requests
                if req.load_spec is not None and req.load_spec.can_load
            )
            saveable = sum(
                1
                for req in metadata.requests
                if req.save_spec is not None and req.save_spec.can_save
            )
            logger.info(
                "[SpecAgent][LMCACHE-TIMING] start_load_kv "
                "result=done metadata_reqs=%d loadable=%d saveable=%d "
                "specagent_fetch_reqs=%d layerwise=%s elapsed_ms=%.3f",
                len(metadata.requests),
                loadable,
                saveable,
                len(sa_fetch_reqs),
                self.use_layerwise,
                (time.perf_counter() - t0) * 1000.0,
            )

    def record_failed_blocks(
        self,
        request_id: str,
        expected_mask: torch.Tensor,
        ret_mask: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> set[int]:
        """Record block IDs associated with failed load attempts.

        Args:
            request_id: request id from vLLM.
            expected_mask: Boolean tensor indicating which tokens were expected to
                be loaded from LMCache. True means the token should be loaded,
                False means the token is already cached in vLLM and does not need
                to be loaded from LMCache.
            ret_mask: Boolean tensor indicating which tokens were actually
                successfully retrieved from LMCache. True means the token was
                successfully loaded. For example, if 256 tokens are expected to be
                loaded, but only 192 tokens are successfully loaded, then the
                ret_mask will be a tensor of 256 items like [T, T, ..., F, F, ...]
                where the first 192 elements are True and the last 64 elements
                are False.
            slot_mapping: Tensor indicating slot IDs for each token. The block
                ID is computed by dividing the slot ID by the block size.

        Example:
            expected_mask = [F, T, T, T] meaning the 1st is in vLLM cache
            ret_mask = [F, T, F, F] meaning failure from loading the 3rd
            missing_mask = expected_mask & ~ret_mask = [F, F, T, T]
            missing_indices = [2, 3]
            then missing_blocks is calculated from slot_mapping and missing_indices

        Returns:
            set[int]: Set of block IDs that failed to load.
        """

        if expected_mask.numel() == 0:
            return set()

        expected_mask_cpu = expected_mask.to(device="cpu", dtype=torch.bool)
        ret_mask_cpu = ret_mask.to(device="cpu", dtype=torch.bool)

        if ret_mask_cpu.shape[0] != expected_mask_cpu.shape[0]:
            logger.debug("expected_mask_cpu.shape[0] != ret_mask_cpu.shape[0]")
            return set()

        missing_mask = expected_mask_cpu & ~ret_mask_cpu
        if not torch.any(missing_mask):
            return set()

        missing_indices = torch.nonzero(missing_mask, as_tuple=False).view(-1)
        if missing_indices.numel() == 0:
            return set()

        slot_mapping_cpu = slot_mapping.to(device="cpu", dtype=torch.long)
        if slot_mapping_cpu.shape[0] > missing_mask.shape[0]:
            slot_mapping_cpu = slot_mapping_cpu[: missing_mask.shape[0]]

        missing_blocks_tensor = torch.unique(
            slot_mapping_cpu[missing_indices] // self._block_size
        )
        missing_blocks = {int(block.item()) for block in missing_blocks_tensor}

        if not missing_blocks:
            return set()

        logger.warning(
            "Request %s failed to load %d tokens across %d blocks",
            request_id,
            missing_indices.numel(),
            len(missing_blocks),
        )
        return missing_blocks

    @_lmcache_nvtx_annotate
    def wait_for_layer_load(self, layer_name: str) -> None:
        """Blocking until the KV for a specific layer is loaded into vLLM's
        paged buffer.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        requested_layer = _specagent_layer_index(layer_name)
        if requested_layer is None:
            requested_layer = self.current_layer
        requested_layer = max(0, min(int(requested_layer), self.num_layers - 1))

        has_specagent_events = any(
            isinstance(meta, dict)
            and isinstance(meta.get("layer_events"), list)
            for meta in self._specagent_layerwise_meta
        )
        group_size = 1
        group_id = requested_layer
        group_end_layer = requested_layer
        if has_specagent_events:
            group_size = max(
                1,
                min(
                    int(getattr(self, "_specagent_layer_group_size", 1) or 1),
                    self.num_layers,
                ),
            )
            group_id = requested_layer // group_size
            group_start_layer = group_id * group_size
            group_end_layer = min(
                group_start_layer + group_size - 1,
                self.num_layers - 1,
            )
            if group_id in self._specagent_waited_layer_groups:
                return

        target_layer = group_end_layer
        specagent_fetch = self._specagent_has_layerwise_fetch()
        if not specagent_fetch and group_end_layer == self.num_layers - 1:
            target_layer = self.num_layers
        start_layer = self.current_layer
        overlap_debug = _specagent_env_true("SPECAGENT_OVERLAP_DEBUG")
        t0 = time.perf_counter() if overlap_debug else 0.0
        expected_layer_mib = self._specagent_expected_layer_mib()
        advance_layers = max(0, target_layer - start_layer + 1)
        expected_wait_mib = expected_layer_mib * advance_layers
        with record_function_or_nullcontext(
            "lmcache connector: wait_for_layer_load "
            f"layer={requested_layer} group={group_id} "
            f"group_size={group_size} target={target_layer} "
            f"advance_layers={advance_layers} "
            f"expected_layer_mib={expected_layer_mib:.1f} "
            f"expected_wait_mib={expected_wait_mib:.1f}"
        ):
            if specagent_fetch:
                self._specagent_wait_for_async_prefetch(group_end_layer)
            else:
                while (
                    self.layerwise_retrievers
                    and self.current_layer <= target_layer
                ):
                    self._wait_for_next_layer_load()

        # Wait only for the layer being consumed.  A stream-wide wait would also
        # wait for the one-group-ahead prefetch launched above, defeating the
        # overlap we are trying to create.
        current_stream = torch.cuda.current_stream(self.device)
        missing_event_dependency = False
        waited_event = False
        for meta in self._specagent_layerwise_meta:
            if meta is None:
                missing_event_dependency = True
                continue
            events = meta.get("layer_events")
            wait_layer = group_end_layer if has_specagent_events else requested_layer
            event = (
                events[wait_layer]
                if isinstance(events, list) and wait_layer < len(events)
                else None
            )
            if event is None:
                missing_event_dependency = True
                continue
            with record_function_or_nullcontext(
                "lmcache connector: wait_for_layer_load event "
                f"layer={requested_layer} wait_layer={wait_layer} "
                f"group={group_id} expected_layer_mib={expected_layer_mib:.1f}"
            ):
                current_stream.wait_event(event)
            waited_event = True

        try:
            gpu_connector = getattr(self.lmcache_engine, "gpu_connector", None)
            load_stream = getattr(gpu_connector, "load_stream", None)
            if missing_event_dependency and load_stream is not None:
                with record_function_or_nullcontext(
                    "lmcache connector: wait_for_layer_load stream_fallback "
                    f"layer={requested_layer} group={group_id} "
                    f"expected_wait_mib={expected_wait_mib:.1f}"
                ):
                    current_stream.wait_stream(load_stream)
        except Exception:
            logger.exception(
                "[SpecAgent][FETCH-SYNC] failed to synchronize load stream "
                "for layer=%s requested_layer=%d target_layer=%d",
                layer_name,
                requested_layer,
                target_layer,
            )
            raise

        if overlap_debug:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if requested_layer < 4 or requested_layer == self.num_layers - 1:
                logger.info(
                    "[SpecAgent][OVERLAP-WAIT] requested_layer=%d "
                    "target_layer=%d group_size=%d group_end=%d "
                    "advanced=%d start_layer=%d end_layer=%d wait_ms=%.3f "
                    "active_retrievers=%d",
                    requested_layer,
                    target_layer,
                    group_size,
                    group_end_layer,
                    max(0, self.current_layer - start_layer),
                    start_layer,
                    self.current_layer,
                    elapsed_ms,
                    len(self.layerwise_retrievers),
                )
            if missing_event_dependency and waited_event:
                logger.warning(
                    "[SpecAgent][OVERLAP-WAIT] layer=%d used_event_wait=True "
                    "fallback_stream_wait=True",
                    requested_layer,
                )

        if has_specagent_events:
            self._specagent_waited_layer_groups.add(group_id)

        return

    def _specagent_kv_bytes_per_layer_token(self) -> int:
        if not self.kv_caches:
            return 0
        kv_layer = next(iter(self.kv_caches.values()))
        if kv_layer.ndim < 5:
            return 0
        return (
            2
            * int(kv_layer.shape[-2])
            * int(kv_layer.shape[-1])
            * int(kv_layer.element_size())
        )

    def _specagent_expected_layer_mib(self) -> float:
        bytes_per_layer_token = self._specagent_kv_bytes_per_layer_token()
        if bytes_per_layer_token <= 0:
            return 0.0
        expected_tokens = sum(
            int(meta.get("expected_tokens", 0) or 0)
            for meta in self._specagent_layerwise_meta
            if isinstance(meta, dict)
        )
        return expected_tokens * bytes_per_layer_token / (1024 * 1024)

    def _specagent_has_layerwise_fetch(self) -> bool:
        return any(
            isinstance(meta, dict) and meta.get("is_specagent")
            for meta in self._specagent_layerwise_meta
        )

    def _specagent_all_fetches_paged_restore(self) -> bool:
        specagent_metas = [
            meta for meta in self._specagent_layerwise_meta
            if isinstance(meta, dict) and meta.get("is_specagent")
        ]
        return bool(specagent_metas) and all(
            meta.get("is_paged_restore") for meta in specagent_metas
        )

    def _specagent_events_ready(self, target_layer: int) -> bool:
        if target_layer < 0:
            return True
        for meta in self._specagent_layerwise_meta:
            events = meta.get("layer_events") if isinstance(meta, dict) else None
            if (
                isinstance(events, list)
                and target_layer < len(events)
                and events[target_layer] is None
            ):
                return False
        return True

    def _specagent_advance_retrievers_until(
        self,
        target_layer: int,
        stream_wait_event: object | None = None,
    ) -> None:
        if target_layer < 0:
            return
        target_layer = min(int(target_layer), self.num_layers - 1)
        load_stream = getattr(
            getattr(self.lmcache_engine, "gpu_connector", None),
            "load_stream",
            None,
        )
        if load_stream is not None:
            try:
                torch.cuda.set_device(load_stream.device)
            except Exception:
                pass

        def validate_result(meta: object, ret_token_mask: object | None) -> None:
            if (
                not isinstance(meta, dict)
                or not meta.get("is_specagent")
                or ret_token_mask is None
            ):
                return
            if hasattr(ret_token_mask, "sum"):
                fetched = int(ret_token_mask.sum().item())
            elif hasattr(ret_token_mask, "item"):
                fetched = int(ret_token_mask.item())
            else:
                fetched = int(ret_token_mask)
            expected = int(
                meta.get("expected_fetch_tokens")
                or meta.get("expected_tokens")
                or 0
            )
            if fetched < expected:
                raise RuntimeError(
                    "[SpecAgent][FETCH-ERROR] retrieved fewer tokens "
                    f"than expected for req={meta.get('req_id')}: "
                    f"fetched={fetched} expected={expected}"
                )

        gpu_connector = getattr(self.lmcache_engine, "gpu_connector", None)
        flush_deferred = getattr(
            gpu_connector, "specagent_flush_deferred_layer", None
        )
        has_deferred = getattr(
            gpu_connector, "specagent_has_deferred_layer", None
        )

        # Advance layer-major across retrievers.  With multiple SpecAgent
        # segments, draining retriever 0 through layer 63 before touching
        # retriever 1 can make layer 0 attention wait behind unrelated later
        # layers.  Interleaving by layer keeps the event needed by the consumer
        # near the front of the producer queue while preserving each retriever's
        # internal layer order.
        scan_layer = min(
            max(0, int(self._specagent_prefetch_scan_layer)),
            target_layer + 1,
        )
        for layer_id in range(scan_layer, target_layer + 1):
            for idx, layerwise_retriever in enumerate(self.layerwise_retrievers):
                meta = (
                    self._specagent_layerwise_meta[idx]
                    if idx < len(self._specagent_layerwise_meta)
                    else {}
                )
                events = (
                    meta.get("layer_events") if isinstance(meta, dict) else None
                )
                is_deferred = bool(
                    isinstance(meta, dict)
                    and meta.get("defer_layer_submit")
                    and callable(has_deferred)
                )
                guard = 0
                while (
                    isinstance(events, list)
                    and layer_id < len(events)
                    and (
                        not has_deferred(layer_id, events)
                        if is_deferred
                        else events[layer_id] is None
                    )
                    and guard < self.num_layers + 4
                ):
                    try:
                        with record_function_or_nullcontext(
                            "lmcache adapter: advance to layer="
                            f"{layer_id} target={target_layer} retriever={idx}"
                        ):
                            ret_token_mask = next(layerwise_retriever)
                    except StopIteration:
                        break
                    guard += 1
                    with self._specagent_prefetch_cond:
                        self._specagent_prefetch_cond.notify_all()
                    validate_result(meta, ret_token_mask)
            if callable(flush_deferred):
                did_flush = flush_deferred(
                    layer_id,
                    scratch_free_event=stream_wait_event,
                )
                if did_flush:
                    # One wait in the load stream orders this and every later
                    # fused scatter in the same advance call.
                    stream_wait_event = None
            if not self._specagent_events_ready(layer_id):
                # Do not skip an incomplete layer.  This is only expected for
                # a non-deferred compatibility retriever that yielded without
                # publishing its event.
                break
            self._specagent_prefetch_scan_layer = layer_id + 1
        with self._specagent_prefetch_cond:
            self._specagent_prefetch_cond.notify_all()

    def _specagent_clear_done_prefetch_future(self) -> None:
        with self._specagent_prefetch_lock:
            future = self._specagent_prefetch_future
            if future is None or not future.done():
                return
            self._specagent_prefetch_future = None
            self._specagent_prefetch_target = -1
        future.result()

    def _specagent_join_prefetch_future(self) -> None:
        with self._specagent_prefetch_lock:
            future = self._specagent_prefetch_future
        if future is None:
            return
        future.result()
        with self._specagent_prefetch_lock:
            if self._specagent_prefetch_future is future:
                self._specagent_prefetch_future = None
                self._specagent_prefetch_target = -1

    def _specagent_run_save_step(
        self,
        req_id: str,
        layer_name: str,
        layerwise_storer: Generator[Optional[torch.Tensor], None, None],
    ) -> tuple[str, bool]:
        try:
            torch.cuda.set_device(self.device)
        except Exception:
            pass
        stopped = False
        with record_function_or_nullcontext(
            "lmcache save worker: next "
            f"req={req_id} layer={layer_name}"
        ):
            try:
                next(layerwise_storer)
            except StopIteration:
                stopped = True
        return req_id, stopped

    def _specagent_submit_save_step(
        self,
        req_id: str,
        layer_name: str,
        layerwise_storer: Generator[Optional[torch.Tensor], None, None],
    ) -> None:
        executor = self._specagent_save_executor
        if executor is None:
            next(layerwise_storer)
            return
        with record_function_or_nullcontext(
            "lmcache connector: submit_save_kv_layer "
            f"req={req_id} layer={layer_name}"
        ):
            future = executor.submit(
                self._specagent_run_save_step,
                req_id,
                layer_name,
                layerwise_storer,
            )
        self._specagent_save_futures.append(future)

    def _specagent_join_save_futures(self) -> None:
        futures = self._specagent_save_futures
        if not futures:
            return
        self._specagent_save_futures = []
        with record_function_or_nullcontext(
            f"lmcache connector: join_save_futures count={len(futures)}"
        ):
            for future in futures:
                req_id, stopped = future.result()
                if stopped:
                    self._layerwise_save_storers.pop(req_id, None)

    def _specagent_advance_after_future(
        self, previous: Future, target_layer: int
    ) -> None:
        with record_function_or_nullcontext(
            "lmcache prefetch worker: wait_prev "
            f"target={target_layer}"
        ):
            previous.result()
        with record_function_or_nullcontext(
            "lmcache prefetch worker: advance "
            f"target={target_layer}"
        ):
            self._specagent_advance_retrievers_until(target_layer)

    def _specagent_wait_event_then_advance(
        self,
        previous: Future | None,
        target_layer: int,
        stream_wait_event: object | None,
    ) -> None:
        if previous is not None:
            with record_function_or_nullcontext(
                "lmcache prefetch worker: wait_prev "
                f"target={target_layer}"
            ):
                previous.result()
        all_deferred = bool(self._specagent_layerwise_meta) and all(
            isinstance(meta, dict) and meta.get("defer_layer_submit")
            for meta in self._specagent_layerwise_meta
        )
        if stream_wait_event is not None and not all_deferred:
            load_stream = getattr(
                getattr(self.lmcache_engine, "gpu_connector", None),
                "load_stream",
                None,
            )
            if load_stream is not None:
                try:
                    torch.cuda.set_device(load_stream.device)
                except Exception:
                    pass
                with record_function_or_nullcontext(
                    "lmcache prefetch worker: stream_wait_event "
                    f"target={target_layer}"
                ):
                    load_stream.wait_event(stream_wait_event)
        with record_function_or_nullcontext(
            "lmcache prefetch worker: advance "
            f"target={target_layer}"
        ):
            self._specagent_advance_retrievers_until(
                target_layer,
                stream_wait_event=(
                    stream_wait_event if all_deferred else None
                ),
            )

    def _specagent_request_async_prefetch(
        self,
        target_layer: int,
        stream_wait_event: object | None = None,
    ) -> Future | None:
        if target_layer < 0:
            return None
        all_deferred = bool(self._specagent_layerwise_meta) and all(
            isinstance(meta, dict) and meta.get("defer_layer_submit")
            for meta in self._specagent_layerwise_meta
        )
        if all_deferred:
            # torch.cuda.Event.record() from this background executor can
            # intermittently stop forever immediately after
            # cudaStreamIsCapturing when CUDA graphs are enabled.  The producer
            # work here only prepares descriptors and enqueues H2D/scatter on
            # load_stream; running that host enqueue on the model thread does
            # not serialize the GPU streams.  It also removes executor/GIL
            # submission gaps from the layer pipeline.
            with record_function_or_nullcontext(
                "lmcache prefetch request: inline_deferred_advance "
                f"target={target_layer}"
            ):
                self._specagent_advance_retrievers_until(
                    target_layer,
                    stream_wait_event=stream_wait_event,
                )
            return None
        executor = self._specagent_prefetch_executor
        if executor is None:
            if stream_wait_event is not None and not all_deferred:
                load_stream = getattr(
                    getattr(self.lmcache_engine, "gpu_connector", None),
                    "load_stream",
                    None,
                )
                if load_stream is not None:
                    try:
                        torch.cuda.set_device(load_stream.device)
                    except Exception:
                        pass
                    with record_function_or_nullcontext(
                        "lmcache prefetch request: inline_stream_wait_event "
                        f"target={target_layer}"
                    ):
                        load_stream.wait_event(stream_wait_event)
            with record_function_or_nullcontext(
                "lmcache prefetch request: inline_advance "
                f"target={target_layer}"
            ):
                self._specagent_advance_retrievers_until(
                    target_layer,
                    stream_wait_event=(
                        stream_wait_event if all_deferred else None
                    ),
                )
            return None
        with record_function_or_nullcontext(
            "lmcache prefetch request: lock "
            f"target={target_layer} event={stream_wait_event is not None}"
        ):
            self._specagent_prefetch_lock.acquire()
        try:
            future = self._specagent_prefetch_future
            if future is not None:
                if future.done():
                    self._specagent_prefetch_future = None
                    self._specagent_prefetch_target = -1
                    with record_function_or_nullcontext(
                        "lmcache prefetch request: done_result "
                        f"target={target_layer}"
                    ):
                        future.result()
                elif self._specagent_prefetch_target >= target_layer:
                    with record_function_or_nullcontext(
                        "lmcache prefetch request: return_existing "
                        f"target={target_layer} current="
                        f"{self._specagent_prefetch_target}"
                    ):
                        pass
                    return future
                else:
                    self._specagent_prefetch_target = target_layer
                    if stream_wait_event is not None:
                        with record_function_or_nullcontext(
                            "lmcache prefetch request: submit_after_event "
                            f"target={target_layer}"
                        ):
                            self._specagent_prefetch_future = executor.submit(
                                self._specagent_wait_event_then_advance,
                                future,
                                target_layer,
                                stream_wait_event,
                            )
                    else:
                        with record_function_or_nullcontext(
                            "lmcache prefetch request: submit_after "
                            f"target={target_layer}"
                        ):
                            self._specagent_prefetch_future = executor.submit(
                                self._specagent_advance_after_future,
                                future,
                                target_layer,
                            )
                    return self._specagent_prefetch_future
            self._specagent_prefetch_target = target_layer
            if stream_wait_event is not None:
                with record_function_or_nullcontext(
                    "lmcache prefetch request: submit_new_event "
                    f"target={target_layer}"
                ):
                    self._specagent_prefetch_future = executor.submit(
                        self._specagent_wait_event_then_advance,
                        None,
                        target_layer,
                        stream_wait_event,
                    )
            else:
                with record_function_or_nullcontext(
                    "lmcache prefetch request: submit_new "
                    f"target={target_layer}"
                ):
                    self._specagent_prefetch_future = executor.submit(
                        self._specagent_advance_retrievers_until,
                        target_layer,
                    )
            return self._specagent_prefetch_future
        finally:
            self._specagent_prefetch_lock.release()

    def _specagent_wait_for_async_prefetch(self, target_layer: int) -> None:
        if self._specagent_events_ready(target_layer):
            self._specagent_clear_done_prefetch_future()
            return
        group_size = max(
            1,
            min(
                int(getattr(self, "_specagent_layer_group_size", 1) or 1),
                self.num_layers,
            ),
        )
        initial_ring_end = min(2 * group_size - 1, self.num_layers - 1)
        reuse_event = None
        if target_layer > initial_ring_end:
            reuse_event = _specagent_record_cuda_event(
                torch.cuda.current_stream(self.device)
            )
        future = self._specagent_request_async_prefetch(
            target_layer,
            stream_wait_event=reuse_event,
        )
        if future is None:
            return
        while not self._specagent_events_ready(target_layer):
            if future.done():
                future.result()
                break
            with self._specagent_prefetch_cond:
                if self._specagent_events_ready(target_layer):
                    break
                self._specagent_prefetch_cond.wait(timeout=0.001)
        if future.done():
            with self._specagent_prefetch_lock:
                if self._specagent_prefetch_future is future:
                    self._specagent_prefetch_future = None
                    self._specagent_prefetch_target = -1
        if not self._specagent_events_ready(target_layer):
            self._specagent_advance_retrievers_until(target_layer)

    def prefetch_next_layer_load(
        self,
        layer_name: str,
        attention_done_event: object | None = None,
    ) -> None:
        """Advance the SpecAgent layerwise producer ahead of the consumer."""
        if (
            not self.layerwise_retrievers
            or not self._specagent_has_layerwise_fetch()
        ):
            return
        requested_layer = _specagent_layer_index(layer_name)
        if requested_layer is None:
            requested_layer = self.current_layer
        requested_layer = max(0, min(int(requested_layer), self.num_layers - 1))
        if self._specagent_all_fetches_paged_restore():
            self._specagent_request_async_prefetch(self.num_layers - 1)
            return

        group_size = max(
            1,
            min(
                int(getattr(self, "_specagent_layer_group_size", 1) or 1),
                self.num_layers,
            ),
        )
        group_id = requested_layer // group_size
        group_end_layer = min(
            (group_id + 1) * group_size - 1,
            self.num_layers - 1,
        )
        if (
            attention_done_event is not None
            and requested_layer < group_end_layer
        ):
            # A grouped scratch slot is not reusable until the last attention
            # in the group has consumed it.  Earlier layers therefore cannot
            # advance the parity-ring producer; avoiding their no-op requests
            # also removes one Python callback from every non-boundary layer.
            return
        next_group_end_layer = min(
            (group_id + 2) * group_size - 1,
            self.num_layers - 1,
        )
        target_layer = next_group_end_layer
        stream_wait_event = None
        if attention_done_event is None:
            # Only the two initially free parity slots may be populated
            # without a scratch-read completion dependency.
            target_layer = min(
                target_layer,
                2 * group_size - 1,
                self.num_layers - 1,
            )
        elif requested_layer >= group_end_layer:
            # For group G, the expression below ends at G+2 when group_size=1:
            # (G + 3) * 1 - 1.  That is exactly the parity slot released by
            # attention G; do not reduce this to ``G + 2 - 1``.
            reuse_group_end_layer = min(
                (group_id + 3) * group_size - 1,
                self.num_layers - 1,
            )
            if reuse_group_end_layer > target_layer:
                target_layer = reuse_group_end_layer
                stream_wait_event = attention_done_event

        overlap_debug = _specagent_env_true("SPECAGENT_OVERLAP_DEBUG")
        t0 = time.perf_counter() if overlap_debug else 0.0
        self._specagent_request_async_prefetch(
            target_layer,
            stream_wait_event=stream_wait_event,
        )

        if overlap_debug and requested_layer < 4:
            logger.info(
                "[SpecAgent][OVERLAP-PREFETCH] requested_layer=%d "
                "target_layer=%d group_size=%d group_end=%d "
                "event_wait=%s enqueue_ms=%.3f active_retrievers=%d",
                requested_layer,
                target_layer,
                group_size,
                group_end_layer,
                stream_wait_event is not None,
                (time.perf_counter() - t0) * 1000.0,
                len(self.layerwise_retrievers),
            )

    def _wait_for_next_layer_load(self) -> None:
        if self.layerwise_retrievers:
            logger.debug("Waiting for layer %d to be loaded", self.current_layer)

        # Wait for the layer to be loaded
        active_retrievers = []
        active_meta = []
        for idx, layerwise_retriever in enumerate(self.layerwise_retrievers):
            meta = (
                self._specagent_layerwise_meta[idx]
                if idx < len(self._specagent_layerwise_meta) else None
            )
            next_label = f"lmcache adapter: next layer={self.current_layer}"
            if isinstance(meta, dict):
                expected_tokens = int(meta.get("expected_tokens") or 0)
                expected_mib = (
                    expected_tokens
                    * self._specagent_kv_bytes_per_layer_token()
                    / (1024 * 1024)
                )
                next_label += (
                    f" req={meta.get('req_id')}"
                    f" segment={meta.get('segment')}"
                    f" expected_tokens={expected_tokens}"
                    f" expected_mib={expected_mib:.1f}"
                )
            try:
                with record_function_or_nullcontext(next_label):
                    ret_token_mask = next(layerwise_retriever)
            except StopIteration:
                continue

            if meta is not None:
                fetched_tokens = None
                if ret_token_mask is not None:
                    if hasattr(ret_token_mask, "sum"):
                        fetched_tokens = int(ret_token_mask.sum().item())
                    elif hasattr(ret_token_mask, "item"):
                        fetched_tokens = int(ret_token_mask.item())
                expected_tokens = int(meta.get("expected_tokens") or 0)
                if (
                    fetched_tokens is not None
                    and fetched_tokens < expected_tokens
                ):
                    raise RuntimeError(
                        "[SpecAgent][FETCH-INCOMPLETE] req=%s layer=%d "
                        "segment=%s fetched=%d expected=%d"
                        % (
                            meta.get("req_id"),
                            self.current_layer,
                            meta.get("segment"),
                            fetched_tokens,
                            expected_tokens,
                        )
                    )
                log_fetch_ready = (
                    _specagent_env_true("SPECAGENT_VERBOSE")
                    or _specagent_env_true("SPECAGENT_FETCH_DEBUG")
                    or _specagent_env_true("SPECAGENT_BW_PROFILE")
                )
                should_log_ready = log_fetch_ready and (
                    self.current_layer == 0
                    or self.current_layer == self.num_layers - 1
                )
                if should_log_ready:
                    logger.info(
                        "[SpecAgent][FETCH-READY] req=%s layer=%d "
                        "segment=%s expected_tokens=%d ret_tokens=%s",
                        meta.get("req_id"),
                        self.current_layer,
                        meta.get("segment"),
                        expected_tokens,
                        (
                            "deferred"
                            if fetched_tokens is None else str(fetched_tokens)
                        ),
                    )
                if (
                    meta.get("copy_to_paged")
                    and self.current_layer != self.num_layers - 1
                ):
                    events = meta.get("layer_events")
                    event = (
                        events[self.current_layer]
                        if isinstance(events, list)
                        and self.current_layer < len(events)
                        else None
                    )
                    current_stream = torch.cuda.current_stream()
                    if event is not None:
                        current_stream.wait_event(event)
                    src_blocks = meta.get("src_blocks")
                    dst_blocks = meta.get("dst_blocks")
                    src_kvcaches = meta.get("src_kvcaches")
                    dst_kvcaches = meta.get("dst_kvcaches")
                    if (
                        src_blocks is not None
                        and dst_blocks is not None
                        and src_kvcaches
                        and dst_kvcaches
                    ):
                        layer_idx = self.current_layer
                        current_platform.insert_blocks_to_device(
                            src_kvcaches[layer_idx],
                            dst_kvcaches[layer_idx],
                            src_blocks,
                            dst_blocks,
                        )
                        if isinstance(events, list):
                            copied_event = _specagent_record_cuda_event(
                                current_stream
                            )
                            events[layer_idx] = copied_event
                        if os.environ.get("SPECAGENT_FETCH_DEBUG") and (
                            layer_idx == 0
                            or layer_idx == self.num_layers - 1
                        ):
                            logger.info(
                                "[SpecAgent][PAGED-RESTORE-COPY] req=%s "
                                "layer=%d blocks=%d",
                                meta.get("req_id"),
                                layer_idx,
                                int(src_blocks.numel())
                                if hasattr(src_blocks, "numel") else 0,
                            )

            if self.current_layer == self.num_layers - 1:
                final_ret = ret_token_mask
                while True:
                    try:
                        next_ret = next(layerwise_retriever)
                    except StopIteration:
                        break
                    if next_ret is not None:
                        final_ret = next_ret

                if meta is not None and final_ret is not None:
                    if hasattr(final_ret, "sum"):
                        final_tokens = int(final_ret.sum().item())
                    elif hasattr(final_ret, "item"):
                        final_tokens = int(final_ret.item())
                    else:
                        final_tokens = int(final_ret)
                    expected_tokens = int(meta.get("expected_tokens") or 0)
                    if final_tokens < expected_tokens:
                        raise RuntimeError(
                            "[SpecAgent][FETCH-INCOMPLETE] req=%s layer=%d "
                            "segment=%s final_fetched=%d expected=%d"
                            % (
                                meta.get("req_id"),
                                self.current_layer,
                                meta.get("segment"),
                                final_tokens,
                                expected_tokens,
                            )
                        )
                    if (
                        _specagent_env_true("SPECAGENT_VERBOSE")
                        or _specagent_env_true("SPECAGENT_FETCH_DEBUG")
                        or _specagent_env_true("SPECAGENT_BW_PROFILE")
                    ):
                        logger.info(
                            "[SpecAgent][FETCH-READY] req=%s layer=%d "
                            "segment=%s expected_tokens=%d ret_tokens=%d "
                            "final=True",
                            meta.get("req_id"),
                            self.current_layer,
                            meta.get("segment"),
                            expected_tokens,
                            final_tokens,
                        )
                    if meta.get("copy_to_paged"):
                        src_blocks = meta.get("src_blocks")
                        dst_blocks = meta.get("dst_blocks")
                        src_kvcaches = meta.get("src_kvcaches")
                        dst_kvcaches = meta.get("dst_kvcaches")
                        if (
                            src_blocks is not None
                            and dst_blocks is not None
                            and src_kvcaches
                            and dst_kvcaches
                        ):
                            layer_idx = self.current_layer
                            current_platform.insert_blocks_to_device(
                                src_kvcaches[layer_idx],
                                dst_kvcaches[layer_idx],
                                src_blocks,
                                dst_blocks,
                            )
                            events = meta.get("layer_events")
                            if isinstance(events, list):
                                copied_event = _specagent_record_cuda_event(
                                    torch.cuda.current_stream()
                                )
                                events[layer_idx] = copied_event
                            if os.environ.get("SPECAGENT_FETCH_DEBUG"):
                                logger.info(
                                    "[SpecAgent][PAGED-RESTORE-COPY] req=%s "
                                    "layer=%d blocks=%d final=True",
                                    meta.get("req_id"),
                                    layer_idx,
                                    int(src_blocks.numel())
                                    if hasattr(src_blocks, "numel") else 0,
                                )
                    events = meta.get("layer_events")
                    event = (
                        events[self.current_layer]
                        if isinstance(events, list)
                        and self.current_layer < len(events)
                        else None
                    )
                    if event is not None:
                        torch.cuda.current_stream(self.device).wait_event(event)
                    else:
                        gpu_connector = getattr(
                            self.lmcache_engine, "gpu_connector", None)
                        load_stream = getattr(gpu_connector, "load_stream", None)
                        if load_stream is not None:
                            torch.cuda.current_stream(
                                self.device).wait_stream(load_stream)
                num_retrieved_tokens = (
                    final_ret.sum().item()
                    if final_ret is not None and hasattr(final_ret, "sum") else 0
                )
                if final_ret is not None and not hasattr(final_ret, "sum"):
                    num_retrieved_tokens = int(final_ret)
                if _specagent_env_true("SPECAGENT_BW_PROFILE"):
                    logger.info("Retrieved %d tokens", num_retrieved_tokens)
                continue
            active_retrievers.append(layerwise_retriever)
            active_meta.append(meta)

        self.layerwise_retrievers = active_retrievers
        self._specagent_layerwise_meta = active_meta

        if self.layerwise_retrievers:
            self.current_layer += 1

        return

    @_lmcache_nvtx_annotate
    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        """Start saving the a layer of KV cache from vLLM's paged buffer
        to the connector.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        assert self.lmcache_engine is not None
        timing = _specagent_timing_enabled()
        t0 = time.perf_counter() if timing else 0.0
        saved_reqs = 0
        saved_tokens = 0

        if not self.use_layerwise:
            return

        if self.kv_role == "kv_consumer":
            # Don't do save if the role is kv_consumer
            return
        if self._parent._connector_metadata is None:
            logger.warning(
                "In connector.save_kv_layer, but the connector metadata is None"
            )
            return
        connector_metadata = self._parent._get_connector_metadata()
        assert isinstance(connector_metadata, LMCacheConnectorMetadata)

        assert len(self.kv_caches) > 0

        kvcaches = list(self.kv_caches.values())
        is_first = True
        layer_idx = _specagent_layer_index(layer_name)
        fp_enabled = (
            _specagent_fp_enabled(layer_idx)
            and os.environ.get("SPECAGENT_KV_FINGERPRINT_PRESTORE")
        )
        fp_store = _specagent_kv_fp_store() if fp_enabled else None

        for request in connector_metadata.requests:
            save_spec = request.save_spec
            specagent_save_debug = (
                _specagent_env_true("SPECAGENT_SAVE_DEBUG")
                and request.load_spec is not None
                and request.load_spec.can_load
            )
            if (
                save_spec is None or not save_spec.can_save
            ) and self.kv_role != "kv_producer":
                if specagent_save_debug:
                    logger.warning(
                        "[SpecAgent][SAVE-LAYER-SKIP] req=%s layer=%s "
                        "reason=no_save_spec can_save=%s meta_tokens=%d",
                        request.req_id,
                        layer_name,
                        save_spec.can_save if save_spec is not None else None,
                        len(request.token_ids),
                    )
                continue

            # The generator created on layer 0 owns the immutable token,
            # slot-mapping and store-mask state for the whole layerwise save.
            # Rebuilding those tensors (including slot_mapping.to(cuda)) on
            # every subsequent model layer used to add roughly 1--2 ms of
            # model-thread work per layer.  Once the generator exists, only
            # advance it.
            layerwise_storer = self._layerwise_save_storers.get(request.req_id)
            if layerwise_storer is not None:
                if self._specagent_async_save_this_step:
                    self._specagent_submit_save_step(
                        request.req_id,
                        layer_name,
                        layerwise_storer,
                    )
                else:
                    try:
                        next(layerwise_storer)
                    except StopIteration:
                        self._layerwise_save_storers.pop(request.req_id, None)
                        if request.is_finished_save:
                            self._specagent_finished_save_completed.add(
                                request.req_id
                            )
                continue

            token_ids = request.token_ids
            assert isinstance(token_ids, list)

            slot_mapping = request.slot_mapping
            assert isinstance(slot_mapping, torch.Tensor)
            assert len(slot_mapping) == len(token_ids)

            # TODO: have a pre-allocated buffer to hold the slot_mappings
            slot_mapping = slot_mapping.to(self.device)

            if self.kv_role == "kv_producer":
                skip_leading_tokens = 0
            else:
                assert save_spec is not None
                skip_leading_tokens = save_spec.skip_leading_tokens

                if skip_leading_tokens >= len(token_ids):
                    if specagent_save_debug:
                        logger.warning(
                            "[SpecAgent][SAVE-LAYER-SKIP] req=%s layer=%s "
                            "reason=skip_ge_len skip=%d token_len=%d",
                            request.req_id,
                            layer_name,
                            skip_leading_tokens,
                            len(token_ids),
                        )
                    continue  # skip this request
                # Align to lmcache chunk size
                skip_leading_tokens = (
                    skip_leading_tokens
                    // self._lmcache_chunk_size
                    * self._lmcache_chunk_size
                )
                if skip_leading_tokens >= len(token_ids):
                    if specagent_save_debug:
                        logger.warning(
                            "[SpecAgent][SAVE-LAYER-SKIP] req=%s layer=%s "
                            "reason=aligned_skip_ge_len skip=%d token_len=%d",
                            request.req_id,
                            layer_name,
                            skip_leading_tokens,
                            len(token_ids),
                        )
                    continue  # skip this request after chunk alignment

            store_mask = torch.ones(len(token_ids), dtype=torch.bool)
            store_mask[:skip_leading_tokens] = False
            if (
                skip_leading_tokens > 0
                and not request.is_finished_save
                and not request.specagent_observation_tail_save
            ):
                aligned_token_len = (
                    len(token_ids)
                    // self._lmcache_chunk_size
                    * self._lmcache_chunk_size
                )
                if aligned_token_len <= skip_leading_tokens:
                    if specagent_save_debug:
                        logger.warning(
                            "[SpecAgent][SAVE-LAYER-SKIP] req=%s layer=%s "
                            "reason=no_aligned_suffix skip=%d token_len=%d "
                            "aligned_len=%d",
                            request.req_id,
                            layer_name,
                            skip_leading_tokens,
                            len(token_ids),
                            aligned_token_len,
                        )
                    continue
                token_ids = token_ids[:aligned_token_len]
                slot_mapping = slot_mapping[:aligned_token_len]
                store_mask = store_mask[:aligned_token_len]

            if specagent_save_debug:
                logger.warning(
                    "[SpecAgent][SAVE-LAYER-START] req=%s layer=%s "
                    "store_tokens=%d total_tokens=%d skip=%d",
                    request.req_id,
                    layer_name,
                    len(token_ids) - skip_leading_tokens,
                    len(token_ids),
                    skip_leading_tokens,
                )

            saved_reqs += 1
            saved_tokens += max(0, len(token_ids) - skip_leading_tokens)
            if (
                layer_idx == 0
                and os.environ.get("SPECAGENT_LEGACY_PRESTORE_PROOF")
            ):
                saved_block_store = _specagent_cpu_saved_block_store()
                saved_chunk_store = _specagent_cpu_saved_chunk_store()
                block_start = skip_leading_tokens // self._block_size
                aligned_saved_len = (
                    len(token_ids)
                    // self._lmcache_chunk_size
                    * self._lmcache_chunk_size
                )
                chunk_start = skip_leading_tokens // self._lmcache_chunk_size
                chunk_end = aligned_saved_len // self._lmcache_chunk_size
                newly_recorded_chunks = 0
                for chunk_i in range(chunk_start, chunk_end):
                    tok_start = chunk_i * self._lmcache_chunk_size
                    tok_end = tok_start + self._lmcache_chunk_size
                    if tok_end > aligned_saved_len:
                        break
                    if not bool(store_mask[tok_start:tok_end].all().item()):
                        continue
                    token_hash = hash(tuple(int(t) for t in token_ids[:tok_end]))
                    key = (tok_end, token_hash)
                    if key not in saved_chunk_store:
                        newly_recorded_chunks += 1
                    saved_chunk_store.add(key)
                block_end = aligned_saved_len // self._block_size
                newly_recorded = 0
                for logical_i in range(block_start, block_end):
                    tok_start = logical_i * self._block_size
                    tok_end = tok_start + self._block_size
                    if tok_end > aligned_saved_len:
                        break
                    if not bool(store_mask[tok_start:tok_end].all().item()):
                        continue
                    token_hash = hash(tuple(int(t) for t in token_ids[:tok_end]))
                    key = (logical_i, token_hash)
                    if key not in saved_block_store:
                        newly_recorded += 1
                    saved_block_store.add(key)
                if (
                    (newly_recorded or newly_recorded_chunks)
                    and _specagent_env_true("SPECAGENT_SAVE_DEBUG")
                ):
                    logger.warning(
                        "[SpecAgent][CPU-SAVED-BLOCKS] req=%s "
                        "new_blocks=%d total_blocks=%d new_chunks=%d "
                        "total_chunks=%d token_range=[%d,%d)",
                        request.req_id,
                        newly_recorded,
                        len(saved_block_store),
                        newly_recorded_chunks,
                        len(saved_chunk_store),
                        skip_leading_tokens,
                        len(token_ids),
                    )

            if fp_enabled and fp_store is not None:
                block_start = skip_leading_tokens // self._block_size
                block_end = cdiv(len(token_ids), self._block_size)
                saved_blocks = 0
                for logical_i in range(block_start, block_end):
                    tok_start = logical_i * self._block_size
                    tok_end = min(tok_start + self._block_size,
                                  len(token_ids))
                    if tok_start >= tok_end:
                        continue
                    token_hash = hash(tuple(
                        int(t) for t in token_ids[:tok_end]
                    ))
                    slots = slot_mapping[tok_start:tok_end].long()
                    phys = torch.div(
                        slots, self._block_size, rounding_mode="floor"
                    )
                    offs = slots % self._block_size
                    kv_tokens = kv_layer[:, phys, offs]
                    fp_key = (layer_idx, logical_i, token_hash)
                    fp_value = (
                        tok_end - tok_start,
                        *_specagent_kv_fingerprint(
                            kv_tokens[0], kv_tokens[1],
                        ),
                    )
                    old_versions = fp_store.get(fp_key)
                    if old_versions is None:
                        old_versions = []
                    elif isinstance(old_versions, tuple):
                        old_versions = [old_versions]
                    best_sum_delta = 0.0
                    best_norm_delta = 0.0
                    if old_versions:
                        deltas = [
                            _specagent_kv_fp_delta(old_fp, fp_value)
                            for old_fp in old_versions
                        ]
                        best_sum_delta, best_norm_delta = min(
                            deltas, key=lambda item: (item[0], item[1])
                        )
                        if best_sum_delta > 1e-3 or best_norm_delta > 1e-2:
                            logger.warning(
                                "[SpecAgent][KV-FP-VERSION] req=%s "
                                "layer=%d logical=%d token_range=[%d,%d) "
                                "prev_versions=%d best_sum_delta=%.6g "
                                "best_norm_delta=%.6g",
                                request.req_id,
                                layer_idx,
                                logical_i,
                                tok_start,
                                tok_end,
                                len(old_versions),
                                best_sum_delta,
                                best_norm_delta,
                            )
                    old_versions.append(fp_value)
                    # Cap per-key history so debug metadata stays bounded.
                    if len(old_versions) > 8:
                        old_versions = old_versions[-8:]
                    fp_store[fp_key] = old_versions
                    saved_blocks += 1
                if saved_blocks:
                    logger.info(
                        "[SpecAgent][KV-FP-SAVE] req=%s layer=%d "
                        "blocks=%d token_range=[%d,%d) store_size=%d",
                        request.req_id,
                        layer_idx,
                        saved_blocks,
                        skip_leading_tokens,
                        len(token_ids),
                        len(fp_store),
                    )

            layerwise_storer = self._layerwise_save_storers.get(request.req_id)
            if layerwise_storer is None:
                logger.debug(
                    "Storing KV cache for %d out of %d tokens "
                    "(skip_leading_tokens=%d) for request %s",
                    len(token_ids) - skip_leading_tokens,
                    len(token_ids),
                    skip_leading_tokens,
                    request.req_id,
                )

                # TODO (Jiayi): need to make layerwise storing
                # compatible with disagg spec
                layerwise_storer = self.lmcache_engine.store_layer(
                    token_ids,
                    mask=store_mask,
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping,
                    # token_ids and slot_mapping remain full-sequence aligned.
                    # store_mask selects the suffix chunks to save; the GPU
                    # connector must still index slot_mapping by absolute
                    # token position.
                    offset=0,
                    sync=is_first,
                    req_id=request.req_id,
                    specagent_block_size=self._block_size,
                    specagent_save_current_layer_only=True,
                    # Enqueue each layer's D2H behind its producer and publish
                    # CPU cache entries only after one final stream completion.
                    # This removes 46 host synchronizations while preserving
                    # the rule that readers never see partially copied KV.
                    specagent_defer_layerwise_store_commit=True,
                    specagent_tail_merge_prefix_tokens=(
                        request.specagent_tail_merge_prefix_tokens
                    ),
                )
                self._layerwise_save_storers[request.req_id] = layerwise_storer
                if is_first:
                    is_first = False

            if self._specagent_async_save_this_step:
                self._specagent_submit_save_step(
                    request.req_id,
                    layer_name,
                    layerwise_storer,
                )
            else:
                try:
                    next(layerwise_storer)
                except StopIteration:
                    self._layerwise_save_storers.pop(request.req_id, None)
                    if request.is_finished_save:
                        self._specagent_finished_save_completed.add(request.req_id)
                    # FIX: the layerwise store path (used for finished/turn-end
                    # saves that are processed during a forward) never recorded
                    # to the proof ledger -- only the bulk without_forward path
                    # did, and it is skipped here because we mark the req
                    # completed above.  So the ledger stayed nearly empty and the
                    # scheduler's cpu_fetchable coverage massively understated
                    # (-> restore under-fires, preempt frees unsafe blocks).
                    # Record the chunks this store just wrote (skip_leading..len),
                    # with the same (chunk_end, hash(token_ids[:chunk_end])) the
                    # proven-check re-derives.  token_ids == all_token_ids[:limit]
                    # so the hashes match.
                    if (
                        (request.is_finished_save or request.is_preempt_save)
                        and request.save_spec is not None
                        and request.save_spec.can_save
                        and isinstance(request.token_ids, list)
                    ):
                        try:
                            from lmcache.v1.cache_engine import (
                                _specagent_record_actual_cpu_store,
                            )
                            _cs = int(self._lmcache_chunk_size)
                            _tok = request.token_ids
                            _skip = (
                                request.save_spec.skip_leading_tokens // _cs * _cs
                            )
                            _last = len(_tok) // _cs * _cs
                            for _ce in range(_cs, _last + 1, _cs):
                                if _ce > _skip:
                                    _specagent_record_actual_cpu_store(
                                        _tok, 0, _ce, self._block_size
                                    )
                        except Exception:
                            pass

        if timing:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            should_log = (
                saved_reqs > 0
                and (
                    layer_idx in {0, self.num_layers - 1}
                    or elapsed_ms >= 1.0
                )
            )
            if should_log:
                logger.info(
                    "[SpecAgent][LMCACHE-TIMING] save_kv_layer layer=%s "
                    "layer_idx=%s saved_reqs=%d saved_tokens=%d "
                    "elapsed_ms=%.3f",
                    layer_name,
                    layer_idx,
                    saved_reqs,
                    saved_tokens,
                    elapsed_ms,
                )

    @_lmcache_nvtx_annotate
    def wait_for_save(self):
        """Blocking until the KV cache is saved to the connector buffer."""
        timing = _specagent_timing_enabled()
        t0 = time.perf_counter() if timing else 0.0

        connector_metadata = self._parent._get_connector_metadata()
        assert isinstance(connector_metadata, LMCacheConnectorMetadata)

        if self.kv_role == "kv_consumer":
            # Don't do save if the role is kv_consumer
            # But still need to unpin the kv caches according to req_id
            # to balance the pin count from contains()
            assert self.lmcache_engine is not None, (
                "LMCacheEngine must be initialized to unpin requests."
            )
            for request in connector_metadata.requests:
                self.lmcache_engine.lookup_unpin(request.req_id)

            if timing:
                logger.info(
                    "[SpecAgent][LMCACHE-TIMING] wait_for_save "
                    "role=consumer metadata_reqs=%d elapsed_ms=%.3f",
                    len(connector_metadata.requests),
                    (time.perf_counter() - t0) * 1000.0,
                )
            return

        if self.use_layerwise:
            self._specagent_join_save_futures()
            finalized = 0
            for request in connector_metadata.requests:
                layerwise_storer = self._layerwise_save_storers.pop(
                    request.req_id, None
                )
                if layerwise_storer is not None:
                    try:
                        next(layerwise_storer)
                    except StopIteration:
                        pass
                    finalized += 1
                    if request.is_finished_save:
                        self._specagent_finished_save_completed.add(
                            request.req_id
                        )
                # A finished-save whose layerwise storer was never created
                # (decode-only steps -- the common case under
                # save_kv_on_turn_end_only) must stay un-completed here so
                # get_finished()'s no-forward bulk store still runs.
                # Marking it completed unconditionally silently dropped the
                # entire turn-end CPU save (external hit 0-43% vs vanilla 95%).
                # unpin the kv caches according to req_id
                self.lmcache_engine.lookup_unpin(request.req_id)
            if timing:
                logger.info(
                    "[SpecAgent][LMCACHE-TIMING] wait_for_save "
                    "mode=layerwise metadata_reqs=%d finalized=%d "
                    "elapsed_ms=%.3f",
                    len(connector_metadata.requests),
                    finalized,
                    (time.perf_counter() - t0) * 1000.0,
                )
            self._specagent_async_save_this_step = False
            return

        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())

        assert self.lmcache_engine is not None

        for request in connector_metadata.requests:
            # unpin the kv caches according to req_id
            self.lmcache_engine.lookup_unpin(request.req_id)

            save_spec = request.save_spec
            if (
                save_spec is None or not save_spec.can_save
            ) and self.kv_role != "kv_producer":
                continue

            token_ids = request.token_ids

            slot_mapping = request.slot_mapping
            assert isinstance(slot_mapping, torch.Tensor)
            assert len(slot_mapping) == len(token_ids)

            # TODO: have a pre-allocated buffer to hold the slot_mappings
            slot_mapping = slot_mapping.to(self.device)

            skip_leading_tokens = save_spec.skip_leading_tokens
            # shared storage disaggregation will not have a disagg_spec passed in
            if self.kv_role == "kv_producer" and request.disagg_spec:
                skip_leading_tokens = min(
                    skip_leading_tokens, request.disagg_spec.num_transferred_tokens
                )

            if skip_leading_tokens >= len(token_ids):
                continue  # skip this request
            # Align to lmcache chunk size
            skip_leading_tokens = (
                skip_leading_tokens
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )
            if skip_leading_tokens >= len(token_ids):
                continue  # skip this request after chunk alignment

            store_mask = torch.ones(len(token_ids), dtype=torch.bool)
            store_mask[:skip_leading_tokens] = False
            if skip_leading_tokens > 0 and not request.is_finished_save:
                aligned_token_len = (
                    len(token_ids)
                    // self._lmcache_chunk_size
                    * self._lmcache_chunk_size
                )
                if aligned_token_len <= skip_leading_tokens:
                    continue
                token_ids = token_ids[:aligned_token_len]
                slot_mapping = slot_mapping[:aligned_token_len]
                store_mask = store_mask[:aligned_token_len]

            logger.debug(
                "Storing KV cache for %d out of %d tokens "
                "(skip_leading_tokens=%d) for request %s",
                len(token_ids) - skip_leading_tokens,
                len(token_ids),
                skip_leading_tokens,
                request.req_id,
            )

            is_last_prefill = request.is_last_prefill
            if is_last_prefill:
                if request.disagg_spec:
                    request.disagg_spec.is_last_prefill = True
            else:
                if not self.enable_blending:
                    token_len = len(token_ids)
                    aligned_token_len = (
                        token_len // self._lmcache_chunk_size * self._lmcache_chunk_size
                    )
                    token_ids = token_ids[:aligned_token_len]
                    store_mask = store_mask[:aligned_token_len]
                    slot_mapping = slot_mapping[:aligned_token_len]

            self.lmcache_engine.store(
                token_ids,
                mask=store_mask,
                kvcaches=kvcaches,
                slot_mapping=slot_mapping,
                offset=0,
                transfer_spec=request.disagg_spec,
                request_configs=request.request_configs,
                req_id=request.req_id,
            )

            # Update skip_leading_tokens only on last rank to ensure
            # each PP stage stores its own KV cache
            if get_pp_group().is_last_rank:
                # NOTE(Jiayi): We assume all tokens are saved
                save_spec.skip_leading_tokens = len(token_ids)
                if request.disagg_spec:
                    request.disagg_spec.num_transferred_tokens = len(token_ids)

        if timing:
            logger.info(
                "[SpecAgent][LMCACHE-TIMING] wait_for_save "
                "mode=bulk metadata_reqs=%d elapsed_ms=%.3f",
                len(connector_metadata.requests),
                (time.perf_counter() - t0) * 1000.0,
            )

    @_lmcache_nvtx_annotate
    def _specagent_save_finished_meta_without_forward(
        self, request: ReqMeta
    ) -> bool:
        """Store a SpecAgent KV region when no model forward is running.

        Used for both turn-end finished saves (``is_finished_save``) and
        mid-turn preempt saves (``is_preempt_save``).  The store itself is
        identical (synchronous ``store_layer`` from the paged buffer); only the
        caller differs in whether the request is subsequently reported as
        finished_sending.
        """
        if not (request.is_finished_save or request.is_preempt_save):
            return False
        builtins._sa_finsave_call = getattr(builtins, "_sa_finsave_call", 0) + 1
        save_spec = request.save_spec
        if save_spec is None or not save_spec.can_save:
            builtins._sa_finsave_nocansave = getattr(
                builtins, "_sa_finsave_nocansave", 0) + 1
            _fnd = os.environ.get("LMCACHE_FINSAVE2_DEBUG")
            if _fnd:
                try:
                    import builtins as _b
                    _fh = getattr(_b, "_lmc_finsave2_fh", None)
                    if _fh is None:
                        _fh = _b._lmc_finsave2_fh = open(_fnd, "a", buffering=1)
                    _fh.write(
                        "FINSAVE2 req=%s kind=%s reason=NOCANSAVE save_spec=%s\n" % (
                            request.req_id,
                            "preempt" if request.is_preempt_save else "finished",
                            "None" if save_spec is None else
                            ("can_save=%s" % save_spec.can_save)))
                except Exception:
                    pass
            return not request.is_preempt_save
        if self.kv_role == "kv_consumer":
            return not request.is_preempt_save
        assert self.lmcache_engine is not None
        assert len(self.kv_caches) > 0

        token_ids = request.token_ids
        slot_mapping = request.slot_mapping
        assert isinstance(token_ids, list)
        assert isinstance(slot_mapping, torch.Tensor)
        assert len(slot_mapping) == len(token_ids)
        if not token_ids:
            return not request.is_preempt_save

        slot_mapping = slot_mapping.to(self.device)
        skip_leading_tokens = (
            save_spec.skip_leading_tokens
            // self._lmcache_chunk_size
            * self._lmcache_chunk_size
        )
        _fnd = os.environ.get("LMCACHE_FINSAVE2_DEBUG")
        if _fnd:
            try:
                import builtins as _b
                _fh = getattr(_b, "_lmc_finsave2_fh", None)
                if _fh is None:
                    _fh = _b._lmc_finsave2_fh = open(_fnd, "a", buffering=1)
                _reason = ("SKIPALL" if skip_leading_tokens >= len(token_ids)
                           else "STORE")
                _fh.write(
                    "FINSAVE2 req=%s kind=%s can_save=%s skip_raw=%d "
                    "skip_chunked=%d len_tok=%d reason=%s\n" % (
                        request.req_id,
                        "preempt" if request.is_preempt_save else "finished",
                        save_spec.can_save, save_spec.skip_leading_tokens,
                        skip_leading_tokens, len(token_ids), _reason))
            except Exception:
                pass
        if skip_leading_tokens >= len(token_ids):
            builtins._sa_finsave_skipall = getattr(
                builtins, "_sa_finsave_skipall", 0) + 1
            # A preempt save may intentionally be probe-only when the tracker
            # says the whole prefix was saved earlier. The post-call lookup is
            # what validates whether that old prefix still exists.
            return True

        builtins._sa_finsave_store = getattr(
            builtins, "_sa_finsave_store", 0) + 1
        # lmcache 계약: mask 의 False(=건너뛸 앞부분) 개수는 청크 배수여야
        # 한다(청크 키가 prefix[:chunk_end] 해시라 경계에서만 시작 가능).
        # 어긋나면 엔진이 ValueError 로 죽어 런 전체가 날아가므로, 죽이지
        # 말고 숫자를 남긴 뒤 위로 정렬한다(위로 = 조금 덜 저장, 쓰레기
        # 저장 위험 없음).  이 로그가 뜨면 skip 을 만든 쪽이 범인이다.
        _engine_chunk = int(getattr(
            getattr(self.lmcache_engine, "token_database", None),
            "chunk_size", self._lmcache_chunk_size) or self._lmcache_chunk_size)
        if _engine_chunk > 0 and skip_leading_tokens % _engine_chunk:
            _aligned = (
                (skip_leading_tokens + _engine_chunk - 1)
                // _engine_chunk * _engine_chunk
            )
            logger.error(
                "[SpecAgent][SAVE-MASK-MISALIGN] req=%s skip=%d "
                "adapter_chunk=%d engine_chunk=%d len_tok=%d -> %d",
                request.req_id, skip_leading_tokens,
                self._lmcache_chunk_size, _engine_chunk,
                len(token_ids), _aligned,
            )
            skip_leading_tokens = _aligned
            if skip_leading_tokens >= len(token_ids):
                return True
        store_mask = torch.ones(len(token_ids), dtype=torch.bool)
        store_mask[:skip_leading_tokens] = False
        storer = self.lmcache_engine.store_layer(
            token_ids,
            mask=store_mask,
            kvcaches=list(self.kv_caches.values()),
            slot_mapping=slot_mapping,
            offset=0,
            sync=True,
            req_id=request.req_id,
            specagent_block_size=self._block_size,
            specagent_save_current_layer_only=False,
        )
        for _ in storer:
            pass
        return True

    @_lmcache_nvtx_annotate
    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        connector_metadata = self._parent._get_connector_metadata()
        builtins._sa_getfin_call = getattr(builtins, "_sa_getfin_call", 0) + 1
        if not isinstance(connector_metadata, LMCacheConnectorMetadata):
            builtins._sa_getfin_nometa = getattr(
                builtins, "_sa_getfin_nometa", 0) + 1
        if _specagent_env_true("SPECAGENT_SAVE_DEBUG") and (
            builtins._sa_getfin_call % 500 == 0
        ):
            logger.warning(
                "[SpecAgent][GETFIN-COUNTERS] calls=%d nometa=%d reqs=%d "
                "finmatch=%d incompleted=%d wouldcall=%d finsave_call=%d "
                "finsave_nocansave=%d",
                builtins._sa_getfin_call,
                getattr(builtins, "_sa_getfin_nometa", 0),
                getattr(builtins, "_sa_getfin_reqs", 0),
                getattr(builtins, "_sa_getfin_finmatch", 0),
                getattr(builtins, "_sa_getfin_incompleted", 0),
                getattr(builtins, "_sa_getfin_wouldcall", 0),
                getattr(builtins, "_sa_finsave_call", 0),
                getattr(builtins, "_sa_finsave_nocansave", 0),
            )
        if isinstance(connector_metadata, LMCacheConnectorMetadata):
            builtins._sa_getfin_reqs = getattr(
                builtins, "_sa_getfin_reqs", 0) + len(connector_metadata.requests)
            builtins._sa_getfin_finmatch = getattr(
                builtins, "_sa_getfin_finmatch", 0) + sum(
                1 for r in connector_metadata.requests if r.is_finished_save)
            for request in connector_metadata.requests:
                if request.is_finished_save and request.req_id in \
                        self._specagent_finished_save_completed:
                    builtins._sa_getfin_incompleted = getattr(
                        builtins, "_sa_getfin_incompleted", 0) + 1
                if (
                    request.is_finished_save
                    and request.req_id
                    not in self._specagent_finished_save_completed
                ):
                    builtins._sa_getfin_wouldcall = getattr(
                        builtins, "_sa_getfin_wouldcall", 0) + 1
                    if self._specagent_save_finished_meta_without_forward(request):
                        self._specagent_finished_save_completed.add(request.req_id)
                # Mid-turn preempt saves: store synchronously WITHOUT reporting
                # the request as finished_sending (it is still alive; the
                # scheduler frees its blocks itself once the ledger confirms).
                # The meta is emitted once (build_connector_meta pops it), so it
                # is processed exactly once here -- no persistent dedup (which
                # would wrongly block a later re-preempt of the same req_id).
                elif request.is_preempt_save:
                    save_completed = (
                        self._specagent_save_finished_meta_without_forward(
                            request
                        )
                    )
                    if save_completed:
                        save_key = (
                            request.req_id,
                            int(request.specagent_preempt_save_id),
                        )
                        # Store completion alone is not enough: the scheduler's
                        # saved-token tracker may outlive LMCache CPU eviction.
                        # Probe the exact token prefix after publication and
                        # report only the prefix that is recoverable now.
                        assert self.lmcache_engine is not None
                        try:
                            proven_tokens = int(
                                self.lmcache_engine.lookup(
                                    tokens=request.token_ids,
                                    search_range=["LocalCPUBackend"],
                                    lookup_id=request.req_id,
                                    pin=True,
                                    request_configs=request.request_configs,
                                )
                                or 0
                            )
                        except Exception:
                            logger.exception(
                                "[SpecAgent][PREEMPT-SAVE-ACK] coverage "
                                "probe failed req=%s save_id=%d",
                                save_key[0], save_key[1],
                            )
                            continue
                        if proven_tokens < len(request.token_ids):
                            # Do not retain a partial pin when the scheduler
                            # cannot safely evict this request. A full pin stays
                            # live until the normal restore/load path calls
                            # lookup_unpin(req_id).
                            self.lmcache_engine.lookup_unpin(request.req_id)
                        self._specagent_preempt_save_completed[save_key] = (
                            proven_tokens,
                            1,
                        )
                        log_ack = (
                            logger.warning
                            if _specagent_env_true("SPECAGENT_SAVE_DEBUG")
                            else logger.info
                        )
                        log_ack(
                            "[SpecAgent][PREEMPT-SAVE-ACK] req=%s save_id=%d "
                            "proven_tokens=%d requested_tokens=%d",
                            save_key[0], save_key[1], proven_tokens,
                            len(request.token_ids),
                        )
        finished_sending = set(self._specagent_finished_save_completed)
        self._specagent_finished_save_completed.clear()
        return finished_sending or None, None

    def build_connector_worker_meta(
        self,
    ) -> LMCacheSpecAgentWorkerMetadata | None:
        """Return exact save ACKs over the normal worker IPC channel."""
        if not self._specagent_preempt_save_completed:
            return None
        meta = LMCacheSpecAgentWorkerMetadata(
            completed_preempt_saves=self._specagent_preempt_save_completed
        )
        self._specagent_preempt_save_completed = {}
        return meta

    def specagent_update_worker_meta(
        self, meta: object
    ) -> None:
        """Consume aggregated worker ACKs on the scheduler side."""
        if not isinstance(meta, LMCacheSpecAgentWorkerMetadata):
            return
        for save_key, (proven, count) in (
            meta.completed_preempt_saves.items()
        ):
            if count >= self.worker_count:
                self._specagent_preempt_save_acks[save_key] = proven

    def specagent_consume_preempt_save_ack(
        self, req_id: str, save_id: int
    ) -> int | None:
        save_key = (req_id, int(save_id))
        return self._specagent_preempt_save_acks.pop(save_key, None)

    def specagent_commit_preempt_save(
        self, req_id: str, proven_tokens: int
    ) -> None:
        """Commit worker-proven coverage for the ensuing NORMAL restore."""
        self._specagent_preempt_forced_lookup[req_id] = int(proven_tokens)
        logger.info(
            "[SpecAgent][SPEC-FIRST-PREEMPT] committed authoritative "
            "lookup req=%s proven_tokens=%d",
            req_id, int(proven_tokens),
        )
        # Discard a lookup result issued before this synchronous store.  Some
        # lookup-server implementations can still return that stale answer,
        # so get_num_new_matched_tokens also uses the committed coverage
        # directly until update_state_after_alloc consumes it.
        cancel_lookup = getattr(self.lookup_client, "cancel_lookup", None)
        if callable(cancel_lookup):
            cancel_lookup(req_id)
        else:
            clear_lookup = getattr(
                self.lookup_client, "clear_lookup_status", None
            )
            if callable(clear_lookup):
                clear_lookup(req_id)

    def get_block_ids_with_load_errors(self) -> set[int]:
        invalid_blocks = self._invalid_block_ids.copy()
        self._invalid_block_ids.clear()
        return invalid_blocks

    @_lmcache_nvtx_annotate
    def shutdown(self):
        """Shutdown the connector by delegating to LMCacheManager."""
        logger.info("Starting LMCacheConnector shutdown...")
        executor = self._specagent_prefetch_executor
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
            self._specagent_prefetch_executor = None
        cpu_prepare_executor = self._specagent_cpu_prepare_executor
        if cpu_prepare_executor is not None:
            cpu_prepare_executor.shutdown(wait=False, cancel_futures=True)
            self._specagent_cpu_prepare_executor = None
        self._manager.stop_services()

    ###################
    # Scheduler side APIs
    ####################

    @_lmcache_nvtx_annotate
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> Optional[int]:
        """
        Check for external KV cache hit.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        timing = _specagent_timing_enabled()
        t0 = time.perf_counter() if timing else 0.0
        lookup_cached = False

        # Ignore DP attention mock requests
        if request.request_id.startswith("mock_req"):
            return 0
        # to handle preempted requests, we want `get_num_new_matched_tokens` to be
        # idempotent under the condition that `update_state_after_alloc` is NOT called
        # then the two side-effects that must be idempotent are:
        # 1. lookup_client caches a result
        #     uncached in `update_state_after_alloc` if this request can be scheduled
        # 2. cache engine will pin the KV caches for the request
        #     unpinned in `wait_for_save` if this request can be scheduled
        if self.kv_role == "kv_producer" and not hasattr(
            self.lookup_client, "supports_producer_reuse"
        ):
            return 0

        req_id = request.request_id

        # lookup_client is always initialized for scheduler role
        assert self.lookup_client is not None

        nosave_probe = getattr(request, "specagent_nosave_resume_probe", None)
        nosave_live_hit = 0
        if isinstance(nosave_probe, dict):
            nosave_proven = min(
                int(nosave_probe.get("proven_tokens", 0) or 0),
                len(request.all_token_ids),
            )
            live_probe = getattr(
                builtins, "_specagent_lmcache_live_saved_tokens", None
            )
            if nosave_proven > 0 and callable(live_probe):
                nosave_live_hit = int(
                    live_probe(request.all_token_ids, nosave_proven) or 0
                )

        forced_lookup = self._specagent_preempt_forced_lookup.get(req_id)
        if nosave_live_hit > 0:
            num_external_hit_tokens = nosave_live_hit
            lookup_cached = True
            logger.info(
                "[SpecAgent][NOSAVE-PARTIAL-LOOKUP] req=%s "
                "hit_tokens=%d proven_tokens=%d",
                req_id,
                num_external_hit_tokens,
                int(nosave_probe.get("proven_tokens", 0)),
            )
        elif forced_lookup is not None:
            num_external_hit_tokens = min(
                int(forced_lookup), len(request.all_token_ids)
            )
            lookup_cached = True
            logger.info(
                "[SpecAgent][SPEC-FIRST-PREEMPT] authoritative lookup "
                "req=%s hit_tokens=%d",
                req_id, num_external_hit_tokens,
            )
        elif (
            num_external_hit_tokens := self.lookup_client.lookup_cache(lookup_id=req_id)
        ) != -1:
            # -1 means no result cached
            # None or int means ongoing (async) or cached result
            lookup_cached = True
            logger.debug(
                f"Found {num_external_hit_tokens} hit tokens for request"
                f" {req_id} in the lookup cache."
            )
        else:
            logger.debug(f"Looking up cache for the first time for request {req_id}!")
            self._requests_priority[req_id] = getattr(request, "priority", 0)

            # token_ids = request.prompt_token_ids
            # all token ids covers the preemption case
            token_ids = request.all_token_ids

            # If the request has multimodal hashes, apply them to the token ids
            mm_hashes, mm_positions = extract_mm_features(request)
            if mm_hashes and mm_positions:
                # TODO(Jiayi): Optimize this
                token_ids = torch.tensor(request.prompt_token_ids)
                apply_mm_hashes_to_token_ids(token_ids, mm_hashes, mm_positions)
                token_ids = token_ids.tolist()

            request_configs = extract_request_configs(request.sampling_params)
            if self.skip_last_n_tokens > 0:
                token_ids = token_ids[: -self.skip_last_n_tokens]

            num_external_hit_tokens = self.lookup_client.lookup(
                token_ids,
                lookup_id=req_id,
                request_configs=request_configs,
            )

        if num_external_hit_tokens is None:
            logger.debug(
                "Reqid: %s, Total tokens %d, Inference Engine computed tokens: %d, "
                "LMCache hit tokens: None.",
                req_id,
                request.num_tokens,
                num_computed_tokens,
            )
            if timing:
                logger.info(
                    "[SpecAgent][LMCACHE-TIMING] lookup req=%s "
                    "result=pending cached_result=%s total_tokens=%d "
                    "vllm_local_tokens=%d elapsed_ms=%.3f",
                    req_id,
                    lookup_cached,
                    request.num_tokens,
                    num_computed_tokens,
                    (time.perf_counter() - t0) * 1000.0,
                )
            return None

        # When prompt length is divisible by the block size and all
        # blocks are cached, we need to recompute the last token.
        # This will be removed in the future if vLLM's scheduler provides
        # a better support for this case.
        need_to_allocate = num_external_hit_tokens - num_computed_tokens

        # In, full-prompt-hit case, we need to recompute the last token
        if num_external_hit_tokens == request.num_tokens:
            need_to_allocate -= 1

        # Check if hit tokens meet the minimum for retrieve
        # If below minimum, skip retrieve but still record hit tokens
        # for skip_leading_tokens to avoid re-storing existing chunks
        min_retrieve = self.config.min_retrieve_tokens
        below_min_retrieve = min_retrieve > 0 and need_to_allocate < min_retrieve

        if below_min_retrieve:
            logger.info(
                "Reqid: %s, Total tokens %d, Inference Engine computed tokens: %d, "
                "LMCache hit tokens: %d, but need to load: %d < min_retrieve %d, "
                "skip retrieve but record for save skip",
                req_id,
                request.num_tokens,
                num_computed_tokens,
                num_external_hit_tokens,
                max(need_to_allocate, 0),
                min_retrieve,
            )
        else:
            logger.info(
                "Reqid: %s, Total tokens %d, Inference Engine computed tokens: %d, "
                "LMCache hit tokens: %d, need to load: %d",
                req_id,
                request.num_tokens,
                num_computed_tokens,
                num_external_hit_tokens,
                max(need_to_allocate, 0),
            )

        self.load_specs[req_id] = LoadSpec(
            vllm_cached_tokens=num_computed_tokens,
            lmcache_cached_tokens=num_external_hit_tokens,
            can_load=False,
        )

        if below_min_retrieve or need_to_allocate <= 0:
            if timing:
                logger.info(
                    "[SpecAgent][LMCACHE-TIMING] lookup req=%s "
                    "result=no_load cached_result=%s total_tokens=%d "
                    "vllm_local_tokens=%d lmcache_hit_tokens=%d "
                    "need_to_load=%d below_min=%s elapsed_ms=%.3f",
                    req_id,
                    lookup_cached,
                    request.num_tokens,
                    num_computed_tokens,
                    num_external_hit_tokens,
                    max(need_to_allocate, 0),
                    below_min_retrieve,
                    (time.perf_counter() - t0) * 1000.0,
                )
            return 0

        # TODO: Align to vLLM block size. Should test whether it can be removed
        # need_to_allocate = need_to_allocate // self._block_size * \
        #        self._block_size

        if timing:
            logger.info(
                "[SpecAgent][LMCACHE-TIMING] lookup req=%s "
                "result=load cached_result=%s total_tokens=%d "
                "vllm_local_tokens=%d lmcache_hit_tokens=%d "
                "need_to_load=%d below_min=%s elapsed_ms=%.3f",
                req_id,
                lookup_cached,
                request.num_tokens,
                num_computed_tokens,
                num_external_hit_tokens,
                max(need_to_allocate, 0),
                below_min_retrieve,
                (time.perf_counter() - t0) * 1000.0,
            )
        return need_to_allocate

    @_lmcache_nvtx_annotate
    def update_state_after_alloc(self, request: "Request", num_external_tokens: int):
        """
        Update KVConnector state after temporary buffer alloc.

        For SharedStorageConnector, update _request_needs_load
        if the CacheManager this allocated blocks for us.
        """

        timing = _specagent_timing_enabled()

        # Clear local status in lookup client when a new request is
        # successfully scheduled.
        assert self.lookup_client is not None
        self.lookup_client.clear_lookup_status(request.request_id)
        # Consume the override only when this allocation actually admitted an
        # external restore. update_state_after_alloc can also be called for a
        # zero-external allocation before a preempted request reaches its
        # restore admission; dropping it there re-exposes the stale async
        # lookup answer.
        if num_external_tokens > 0:
            forced = self._specagent_preempt_forced_lookup.pop(
                request.request_id, None
            )
            if forced is not None:
                logger.info(
                    "[SpecAgent][SPEC-FIRST-PREEMPT] consumed authoritative "
                    "lookup req=%s proven_tokens=%d external_tokens=%d",
                    request.request_id, forced, num_external_tokens,
                )

        kv_transfer_params = (
            request.kv_transfer_params
            if hasattr(request, "kv_transfer_params")
            else None
        )

        if kv_transfer_params is not None and "disagg_spec" in kv_transfer_params:
            req_disagg_spec = kv_transfer_params["disagg_spec"]

            receiver_id = req_disagg_spec["receiver_host"] + str(
                req_disagg_spec["receiver_init_port"]
            )

            disagg_spec = DisaggSpec(
                req_id=req_disagg_spec["req_id"],
                receiver_id=receiver_id,
                receiver_host=req_disagg_spec["receiver_host"],
                receiver_init_port=req_disagg_spec["receiver_init_port"],
                receiver_alloc_port=req_disagg_spec["receiver_alloc_port"],
            )

            tmp_disagg_tracker[request.request_id] = disagg_spec
        self._unfinished_requests[request.request_id] = request

        if request.request_id not in self.load_specs:
            # No KV tokens from external KV cache, return
            if timing:
                logger.info(
                    "[SpecAgent][LMCACHE-TIMING] after_alloc req=%s "
                    "result=no_load_spec num_external_tokens=%d "
                    "num_computed_tokens=%d total_tokens=%d",
                    request.request_id,
                    num_external_tokens,
                    request.num_computed_tokens,
                    request.num_tokens,
                )
            return

        if num_external_tokens == 0:
            # No need to load anything
            self.load_specs[request.request_id].can_load = False
            if timing:
                spec = self.load_specs[request.request_id]
                logger.info(
                    "[SpecAgent][LMCACHE-TIMING] after_alloc req=%s "
                    "result=no_external_load num_external_tokens=0 "
                    "load_vllm=%d load_lmcache=%d can_load=%s "
                    "num_computed_tokens=%d total_tokens=%d",
                    request.request_id,
                    spec.vllm_cached_tokens,
                    spec.lmcache_cached_tokens,
                    spec.can_load,
                    request.num_computed_tokens,
                    request.num_tokens,
                )
            return

        recalc_last = (
            1
            if (
                self.load_specs[request.request_id].lmcache_cached_tokens
                == request.num_tokens
            )
            else 0
        )
        assert (
            num_external_tokens
            == self.load_specs[request.request_id].lmcache_cached_tokens
            - self.load_specs[request.request_id].vllm_cached_tokens
            - recalc_last
        ), (
            f"Mismatch in tokens to load: {num_external_tokens} vs "
            f"{self.load_specs[request.request_id].lmcache_cached_tokens} "
            "(tokens in lmcache) - "
            f"{self.load_specs[request.request_id].vllm_cached_tokens} "
            "(tokens in vllm) - "
            f"{recalc_last} "
            "(full lmcache hits subtracts last token to recalculate logits)"
            f" for request {request.request_id}"
        )

        self.load_specs[request.request_id].can_load = True
        if timing:
            spec = self.load_specs[request.request_id]
            logger.info(
                "[SpecAgent][LMCACHE-TIMING] after_alloc req=%s "
                "result=external_load num_external_tokens=%d "
                "load_vllm=%d load_lmcache=%d can_load=%s "
                "num_computed_tokens=%d total_tokens=%d",
                request.request_id,
                num_external_tokens,
                spec.vllm_cached_tokens,
                spec.lmcache_cached_tokens,
                spec.can_load,
                request.num_computed_tokens,
                request.num_tokens,
            )

    @_lmcache_nvtx_annotate
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """Attach the connector metadata to the request object.

        This function should NOT modify other fields in the scheduler_output
        except the `kv_connector_metadata` field.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """

        force_skip_save = self.kv_role == "kv_consumer" or self.force_skip_save

        meta = LMCacheConnectorMetadata()
        specagent_fetch_reqs = (
            getattr(scheduler_output, "specagent_fetch_reqs", None) or {}
        )
        specagent_token_counts = (
            getattr(scheduler_output, "specagent_fetch_token_counts", None) or {}
        )
        specagent_cpu_prefix_blocks = (
            getattr(scheduler_output, "specagent_fetch_cpu_prefix_blocks", None)
            or {}
        )
        specagent_token_ranges = (
            getattr(scheduler_output, "specagent_fetch_token_ranges", None) or {}
        )
        specagent_fetch_resident_logical_indices = (
            getattr(
                scheduler_output,
                "specagent_fetch_resident_logical_indices",
                None,
            )
            or {}
        )
        specagent_skip_save_reqs = (
            getattr(scheduler_output, "specagent_skip_save_reqs", None) or {}
        )
        specagent_turn_end_save_reqs = set(
            getattr(scheduler_output, "specagent_turn_end_save_reqs", None) or ()
        )
        specagent_current_block_ids = (
            getattr(scheduler_output, "specagent_current_block_ids", None) or {}
        )
        emitted_req_ids: set[str] = set()
        specagent_scratch_safe_suffix_starts: dict[str, int] = {}
        specagent_scratch_merge_prefix_starts: dict[str, int] = {}

        def flatten_block_ids(
            block_ids: Union[tuple[list[int], ...], list[int], None],
        ) -> list[int] | None:
            if block_ids is None:
                return None
            if isinstance(block_ids, tuple):
                if len(block_ids) == 0:
                    return []
                return list(block_ids[0])
            if isinstance(block_ids, list):
                return [
                    int(i)
                    for elem in block_ids
                    for i in (elem if isinstance(elem, list) else [elem])
                ]
            return None

        def sync_authoritative_block_ids(
            request_tracker: RequestTracker,
            req_id: str,
            reason: str,
        ) -> None:
            authoritative = flatten_block_ids(
                specagent_current_block_ids.get(req_id)
            )
            if authoritative is None:
                return
            old = request_tracker.allocated_block_ids
            if old == authoritative:
                return
            if os.environ.get("SPECAGENT_BLOCKMAP_DEBUG"):
                old_preview = old[:8]
                new_preview = authoritative[:8]
                logger.warning(
                    "[SpecAgent][BLOCKMAP-SYNC] req=%s reason=%s "
                    "old_blocks=%d new_blocks=%d old_preview=%s "
                    "new_preview=%s",
                    req_id,
                    reason,
                    len(old),
                    len(authoritative),
                    old_preview,
                    new_preview,
                )
            request_tracker.allocated_block_ids = authoritative

        def specagent_step_skip_save(req_id: str) -> bool:
            # Explicitly skipped requests are draft/polluted paths and must
            # not write KV.  This decision must dominate fetch metadata:
            # a DRAFT request can carry fetch state for sparse attention, but
            # its newly computed K/V is based on draft tokens and is not a
            # valid CPU-backup source.
            if req_id in specagent_skip_save_reqs:
                return True
            # SpecAgent fetch requests are different: their
            # fetched prefix may live in scratch while paged KV still has holes,
            # but any newly computed observation/verified suffix is valid KV.
            # apply_specagent_scratch_save_floor() raises skip_leading_tokens to
            # the first safe suffix position so the prefix is not re-saved from
            # stale paged slots while the new stable suffix can still be backed
            # up to CPU.
            if req_id in specagent_fetch_reqs:
                return False
            return False

        def specagent_step_force_save(req_id: str) -> bool:
            return req_id in specagent_turn_end_save_reqs

        for req_id, req_meta in list(
            getattr(self, "_specagent_finished_save_metas", {}).items()
        ):
            meta.add_request(req_meta)
            emitted_req_ids.add(req_id)
            self._specagent_finished_save_metas.pop(req_id, None)
            if _specagent_env_true("SPECAGENT_SAVE_DEBUG"):
                logger.warning(
                    "[SpecAgent][FINISH-SAVE-META] emit req=%s tokens=%d "
                    "skip=%s",
                    req_id,
                    len(req_meta.token_ids),
                    (
                        req_meta.save_spec.skip_leading_tokens
                        if req_meta.save_spec is not None
                        else None
                    ),
                )

        # Mid-turn preempt saves: emitted like finished saves so the worker can
        # store them without a forward, but flagged is_preempt_save so they are
        # NOT reported as finished_sending (the request stays alive).
        for req_id, req_meta in list(
            getattr(self, "_specagent_preempt_save_metas", {}).items()
        ):
            meta.add_request(req_meta)
            emitted_req_ids.add(req_id)
            self._specagent_preempt_save_metas.pop(req_id, None)
            if _specagent_env_true("SPECAGENT_SAVE_DEBUG"):
                logger.warning(
                    "[SpecAgent][PREEMPT-SAVE-META] emit req=%s tokens=%d "
                    "skip=%s",
                    req_id,
                    len(req_meta.token_ids),
                    (
                        req_meta.save_spec.skip_leading_tokens
                        if req_meta.save_spec is not None
                        else None
                    ),
                )

        def set_step_skip_save(
            request_tracker: RequestTracker, req_id: str
        ) -> bool:
            old_skip_save = request_tracker.skip_save
            if specagent_step_skip_save(req_id):
                request_tracker.skip_save = True
                if _specagent_env_true("SPECAGENT_SAVE_DEBUG"):
                    logger.warning(
                        "[SpecAgent][SAVE-SKIP] req=%s reason=%s",
                        req_id,
                        specagent_skip_save_reqs.get(req_id),
                    )
            return old_skip_save

        def restore_step_skip_save(
            request_tracker: RequestTracker, old_skip_save: bool
        ) -> None:
            request_tracker.skip_save = old_skip_save

        def apply_specagent_scratch_save_floor(
            request_tracker: RequestTracker,
            req_id: str,
            safe_suffix_start: Optional[int] = None,
        ) -> int:
            old_saved_tokens = request_tracker.num_saved_tokens
            if req_id not in specagent_fetch_reqs:
                return old_saved_tokens
            _sf_t0 = time.perf_counter()

            if safe_suffix_start is None:
                # SpecAgent full-attention recovery may consume missing prefix
                # blocks from the scratch buffer while the paged table still has
                # holes for those logical positions. Saving any part of that
                # prefix from paged KV can overwrite the valid CPU copy with
                # hole/stale KV. The only safe save range for this scheduler
                # step starts after the prefix that attention read.
                n_scheduled = int(
                    scheduler_output.num_scheduled_tokens.get(req_id, 0) or 0
                )
                safe_suffix_start = max(
                    0, len(request_tracker.token_ids) - n_scheduled,
                )

            min_save_start = max(old_saved_tokens, safe_suffix_start)
            specagent_scratch_safe_suffix_starts[req_id] = int(
                safe_suffix_start
            )
            save_floor = (
                min_save_start // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )
            merge_observation_tail = (
                specagent_step_force_save(req_id)
                and specagent_fetch_reqs.get(req_id) == "observation_prefill"
            )
            if save_floor < min_save_start:
                boundary_chunk = (save_floor, save_floor + self._lmcache_chunk_size)
                cpu_prefix_blocks = int(
                    specagent_cpu_prefix_blocks.get(req_id, 0) or 0
                )
                resident_groups = (
                    specagent_fetch_resident_logical_indices.get(req_id, ())
                )
                resident_logical: set[int] = set()
                for group in resident_groups:
                    if isinstance(group, (list, tuple, set)):
                        resident_logical.update(int(i) for i in group)
                    else:
                        resident_logical.add(int(group))
                # Only logical blocks intersecting the single boundary chunk
                # can satisfy the overlap predicate; iterating the full
                # cpu_prefix_blocks range (O(context), every fetch req, every
                # step) is equivalent but wasteful.
                overlaps_missing = False
                overlap_lo = max(0, boundary_chunk[0] // self._block_size)
                overlap_hi = min(
                    cpu_prefix_blocks,
                    cdiv(boundary_chunk[1], self._block_size),
                )
                for logical_i in range(overlap_lo, overlap_hi):
                    if logical_i in resident_logical:
                        continue
                    overlaps_missing = True
                    break
                if overlaps_missing:
                    if merge_observation_tail:
                        specagent_scratch_merge_prefix_starts[req_id] = int(
                            safe_suffix_start
                        )
                    else:
                        save_floor = (
                            cdiv(min_save_start, self._lmcache_chunk_size)
                            * self._lmcache_chunk_size
                        )
            if save_floor > old_saved_tokens:
                request_tracker.num_saved_tokens = save_floor
                if _specagent_env_true("SPECAGENT_SAVE_DEBUG"):
                    logger.warning(
                        "[SpecAgent][SAVE-FLOOR] req=%s reason=%s "
                        "old_saved=%d safe_suffix_start=%d save_floor=%d "
                        "token_len=%d scheduled=%d",
                        req_id,
                        specagent_fetch_reqs.get(req_id),
                        old_saved_tokens,
                        safe_suffix_start,
                        save_floor,
                        len(request_tracker.token_ids),
                        int(
                            scheduler_output.num_scheduled_tokens.get(req_id, 0)
                            or 0
                        ),
                    )
            builtins._lmc_ms_save_floor = getattr(
                builtins, "_lmc_ms_save_floor", 0.0) + (
                time.perf_counter() - _sf_t0) * 1000.0
            return old_saved_tokens

        def restore_specagent_scratch_save_floor_if_unsaved(
            request_tracker: RequestTracker,
            req_id: str,
            old_saved_tokens: int,
            req_meta: Optional[ReqMeta],
        ) -> None:
            if req_id not in specagent_fetch_reqs:
                return
            can_save = (
                req_meta is not None
                and req_meta.save_spec is not None
                and req_meta.save_spec.can_save
            )
            if not can_save:
                request_tracker.num_saved_tokens = old_saved_tokens

        def sanitize_specagent_scratch_save_meta(
            req_id: str,
            req_meta: Optional[ReqMeta],
        ) -> Optional[ReqMeta]:
            if req_meta is None:
                return req_meta
            save_spec = req_meta.save_spec
            if save_spec is None or not save_spec.can_save:
                return req_meta
            if specagent_step_force_save(req_id):
                # Observation completion is a stable prefill checkpoint even
                # for ordinary paged/full-context requests. Do not let the
                # layerwise path re-trim its final partial chunk.
                req_meta.specagent_observation_tail_save = True
            if req_id not in specagent_fetch_reqs:
                return req_meta
            merge_observation_tail = (
                specagent_step_force_save(req_id)
                and specagent_fetch_reqs.get(req_id) == "observation_prefill"
            )
            if merge_observation_tail:
                safe_suffix_start = int(
                    specagent_scratch_safe_suffix_starts.get(req_id, 0)
                )
                req_meta.specagent_tail_merge_prefix_tokens = int(
                    specagent_scratch_merge_prefix_starts.get(req_id, 0)
                )
                save_spec.skip_leading_tokens = (
                    safe_suffix_start // self._lmcache_chunk_size
                    * self._lmcache_chunk_size
                )
                return req_meta
            skip_leading_tokens = (
                save_spec.skip_leading_tokens
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )
            aligned_token_len = (
                len(req_meta.token_ids)
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )
            if aligned_token_len <= skip_leading_tokens:
                save_spec.can_save = False
                if _specagent_env_true("SPECAGENT_SAVE_DEBUG"):
                    logger.warning(
                        "[SpecAgent][SAVE-SUFFIX-SKIP] req=%s reason=%s "
                        "skip=%d token_len=%d aligned_len=%d",
                        req_id,
                        specagent_fetch_reqs.get(req_id),
                        skip_leading_tokens,
                        len(req_meta.token_ids),
                        aligned_token_len,
                    )
                return req_meta
            if aligned_token_len < len(req_meta.token_ids):
                req_meta.token_ids = req_meta.token_ids[:aligned_token_len]
                req_meta.slot_mapping = req_meta.slot_mapping[:aligned_token_len]
                if _specagent_env_true("SPECAGENT_SAVE_DEBUG"):
                    logger.warning(
                        "[SpecAgent][SAVE-SUFFIX-TRIM] req=%s reason=%s "
                        "skip=%d token_len=%d aligned_len=%d",
                        req_id,
                        specagent_fetch_reqs.get(req_id),
                        skip_leading_tokens,
                        len(req_meta.token_ids),
                        aligned_token_len,
                    )
            save_spec.skip_leading_tokens = skip_leading_tokens
            return req_meta

        def debug_specagent_req_meta(
            stage: str,
            req_id: str,
            request_tracker: RequestTracker,
            req_meta: Optional[ReqMeta],
            load_spec: Optional[LoadSpec],
        ) -> None:
            if (
                req_id not in specagent_fetch_reqs
                or not _specagent_env_true("SPECAGENT_SAVE_DEBUG")
            ):
                return
            save_spec = req_meta.save_spec if req_meta is not None else None
            logger.warning(
                "[SpecAgent][SAVE-REQMETA] stage=%s req=%s reason=%s "
                "tracker_tokens=%d tracker_saved=%d tracker_skip_save=%s "
                "blocks=%d slot_capacity=%d meta=%s meta_tokens=%d "
                "can_save=%s skip=%s load=%s lmcache_cached=%s vllm_cached=%s",
                stage,
                req_id,
                specagent_fetch_reqs.get(req_id),
                len(request_tracker.token_ids),
                request_tracker.num_saved_tokens,
                request_tracker.skip_save,
                len(request_tracker.allocated_block_ids),
                len(request_tracker.allocated_block_ids) * self._block_size,
                req_meta is not None,
                len(req_meta.token_ids) if req_meta is not None else 0,
                save_spec.can_save if save_spec is not None else None,
                save_spec.skip_leading_tokens if save_spec is not None else None,
                load_spec is not None,
                load_spec.lmcache_cached_tokens if load_spec is not None else None,
                load_spec.vllm_cached_tokens if load_spec is not None else None,
            )

        def get_specagent_load_spec(req_id: str) -> Optional[LoadSpec]:
            if req_id not in specagent_fetch_reqs:
                return None
            # The exact sparse missing ranges are computed later on the worker
            # side in GPUModelRunner._make_specagent_destination_kwargs(), after
            # the final batch/seq/block-table state is known.  At scheduler-side
            # metadata construction time specagent_fetch_token_ranges is usually
            # empty, and treating that as "no missing KV" drops LoadSpec.  Then
            # the worker can discover real holes but LMCache has no request
            # metadata and skips the fetch.  Emit a LoadSpec for every
            # SpecAgent fetch request; the worker-side sparse token mask will
            # still fetch zero tokens when there are no holes.
            cpu_prefix_blocks = int(
                specagent_cpu_prefix_blocks.get(req_id, 0) or 0
            )
            token_count = cpu_prefix_blocks * self._block_size
            if token_count <= 0:
                token_count = int(specagent_token_counts.get(req_id, 0) or 0)
            if token_count <= 0:
                request = self._unfinished_requests.get(req_id)
                if request is not None:
                    token_count = len(request.all_token_ids)
            if token_count <= 0:
                logger.warning(
                    "[SpecAgent][LOADSPEC] skip req=%s reason=no_token_count "
                    "fetch_reason=%s",
                    req_id,
                    specagent_fetch_reqs.get(req_id),
                )
                return None
            logger.debug(
                "[SpecAgent][LOADSPEC] force req=%s fetch_reason=%s "
                "lmcache_cached_tokens=%d",
                req_id,
                specagent_fetch_reqs.get(req_id),
                token_count,
            )
            return LoadSpec(
                vllm_cached_tokens=0,
                lmcache_cached_tokens=token_count,
                can_load=True,
            )

        def emit_missing_specagent_fetch_meta() -> None:
            for req_id in specagent_fetch_reqs:
                if req_id in emitted_req_ids:
                    continue
                load_spec = get_specagent_load_spec(req_id)
                request_tracker = self._request_trackers.get(req_id)
                if load_spec is None or request_tracker is None:
                    if load_spec is not None:
                        logger.warning(
                            "[SpecAgent][LOADSPEC] missing tracker req=%s "
                            "fetch_reason=%s",
                            req_id,
                            specagent_fetch_reqs.get(req_id),
                        )
                    continue
                old_saved_tokens = apply_specagent_scratch_save_floor(
                    request_tracker,
                    req_id,
                    len(request_tracker.token_ids),
                )
                sync_authoritative_block_ids(
                    request_tracker, req_id, "extra_fetch_meta"
                )
                req_meta = ReqMeta.from_request_tracker(
                    request_tracker,
                    self._block_size,
                    self._lmcache_chunk_size,
                    load_spec=load_spec,
                    discard_partial_chunks=self._discard_partial_chunks,
                    save_decode_cache=self.config.save_decode_cache,
                    force_save=specagent_step_force_save(req_id),
                )
                debug_specagent_req_meta(
                    "extra_fetch_meta/from_tracker",
                    req_id,
                    request_tracker,
                    req_meta,
                    load_spec,
                )
                restore_specagent_scratch_save_floor_if_unsaved(
                    request_tracker,
                    req_id,
                    old_saved_tokens,
                    req_meta,
                )
                req_meta = sanitize_specagent_scratch_save_meta(req_id, req_meta)
                debug_specagent_req_meta(
                    "extra_fetch_meta/sanitized",
                    req_id,
                    request_tracker,
                    req_meta,
                    load_spec,
                )
                if req_meta is not None:
                    meta.add_request(req_meta)
                    emitted_req_ids.add(req_id)
                    logger.debug(
                        "[SpecAgent][LOADSPEC] emitted extra req_meta req=%s "
                        "fetch_reason=%s",
                        req_id,
                        specagent_fetch_reqs.get(req_id),
                    )

        for finished_req_id in scheduler_output.finished_req_ids:
            self._request_trackers.pop(finished_req_id, None)
            self._unfinished_requests.pop(finished_req_id, None)

        # We should load KV for:
        # 1. new requests
        # 2. preempted requests (once per recovery)
        # can_load will only be True if `update_state_after_alloc` has been called
        # which only happens when vLLM's KV manager has space to receive KV from LMCache
        for request in scheduler_output.scheduled_new_reqs:
            # Ignore DP attention mock requests
            if request.req_id.startswith("mock_req"):
                continue
            load_spec = self.load_specs.pop(request.req_id, None)
            if request.req_id in specagent_fetch_reqs:
                # SpecAgent scratch fetch is authoritative for this step.  A
                # normal LMCache lookup may have produced a can_load=False
                # LoadSpec when vLLM believes the prefix is already computed,
                # but sparse residency can still have holes that must be
                # fetched before attention.  Override that stale/no-op LoadSpec
                # so the first scheduled forward does not run without data.
                forced_load_spec = get_specagent_load_spec(request.req_id)
                if forced_load_spec is not None:
                    load_spec = forced_load_spec
            elif load_spec is None:
                load_spec = get_specagent_load_spec(request.req_id)
            num_tokens_to_compute = (
                request.num_computed_tokens
                + scheduler_output.num_scheduled_tokens[request.req_id]
            )
            lmcache_cached_tokens = 0
            if load_spec is not None:
                lmcache_cached_tokens = load_spec.lmcache_cached_tokens
            request_priority = self._requests_priority.pop(request.req_id, 0)

            persistent_skip_save = force_skip_save or (
                self.config.priority_limit is not None
                and request_priority > self.config.priority_limit
            )
            skip_save = persistent_skip_save or specagent_step_skip_save(
                request.req_id
            )

            tracker_lmcache_cached_tokens = (
                0 if request.req_id in specagent_fetch_reqs
                else lmcache_cached_tokens
            )
            if request.req_id in specagent_fetch_reqs and lmcache_cached_tokens:
                logger.warning(
                    "[SpecAgent][SAVE-META] req=%s forced_load_tokens=%d "
                    "tracker_saved_tokens_reset=0 reason=scratch_fetch_load_is_not_"
                    "proof_that_cpu_has_full_prefix",
                    request.req_id,
                    lmcache_cached_tokens,
                )

            request_tracker = RequestTracker.from_new_request(
                self.config,
                request,
                num_tokens_to_compute,
                tracker_lmcache_cached_tokens,
                skip_save,
            )
            sync_authoritative_block_ids(
                request_tracker, request.req_id, "new_request"
            )
            self._request_trackers[request.req_id] = request_tracker

            old_saved_tokens = apply_specagent_scratch_save_floor(
                request_tracker,
                request.req_id,
            )
            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                self._lmcache_chunk_size,
                load_spec=load_spec,
                discard_partial_chunks=self._discard_partial_chunks,
                save_decode_cache=self.config.save_decode_cache,
                force_save=specagent_step_force_save(request.req_id),
            )
            debug_specagent_req_meta(
                "new_request/from_tracker",
                request.req_id,
                request_tracker,
                req_meta,
                load_spec,
            )
            restore_specagent_scratch_save_floor_if_unsaved(
                request_tracker,
                request.req_id,
                old_saved_tokens,
                req_meta,
            )
            req_meta = sanitize_specagent_scratch_save_meta(
                request.req_id,
                req_meta,
            )
            debug_specagent_req_meta(
                "new_request/sanitized",
                request.req_id,
                request_tracker,
                req_meta,
                load_spec,
            )
            if specagent_step_skip_save(request.req_id):
                request_tracker.skip_save = persistent_skip_save
            if req_meta is not None:
                meta.add_request(req_meta)
                emitted_req_ids.add(request.req_id)

        cached_reqs = scheduler_output.scheduled_cached_reqs

        # NOTE: For backward compatibility with vllm version < 0.9.2,
        # In the latest vllm version, the type of scheduled_cached_reqs has
        # changed from list to object `CachedRequestData`
        if isinstance(cached_reqs, list):
            for i, req in enumerate(cached_reqs):
                load_spec = self.load_specs.pop(req.req_id, None)
                if req.req_id in specagent_fetch_reqs:
                    forced_load_spec = get_specagent_load_spec(req.req_id)
                    if forced_load_spec is not None:
                        load_spec = forced_load_spec
                elif load_spec is None:
                    load_spec = get_specagent_load_spec(req.req_id)
                lmcache_cached_tokens = 0
                vllm_cached_tokens = 0
                if load_spec is not None:
                    lmcache_cached_tokens = load_spec.lmcache_cached_tokens
                    vllm_cached_tokens = load_spec.vllm_cached_tokens
                request_tracker = self._request_trackers.get(req.req_id)
                if request_tracker is None:
                    logger.warning(
                        "[SpecAgent][TRACKER-MISS] extra-fetch req without "
                        "tracker req=%s -- skipped", req.req_id)
                    continue

                # Pass all_token_ids for preempted requests to restore
                # token_ids correctly for chunk key computation
                all_token_ids = None
                if req.resumed_from_preemption:
                    vllm_request = self._unfinished_requests.get(req.req_id)
                    assert vllm_request is not None, (
                        f"Preempted request {req.req_id} not found "
                        "in _unfinished_requests"
                    )
                    all_token_ids = list(vllm_request.all_token_ids)

                tracker_lmcache_cached_tokens = (
                    request_tracker.num_saved_tokens
                    if req.req_id in specagent_fetch_reqs
                    else lmcache_cached_tokens
                )
                if req.req_id in specagent_fetch_reqs and lmcache_cached_tokens:
                    logger.warning(
                        "[SpecAgent][SAVE-META] req=%s forced_load_tokens=%d "
                        "tracker_saved_tokens_kept=%d",
                        req.req_id,
                        lmcache_cached_tokens,
                        tracker_lmcache_cached_tokens,
                    )

                request_tracker.update(
                    req.new_token_ids,
                    req.new_block_ids,
                    req.resumed_from_preemption,
                    lmcache_cached_tokens=tracker_lmcache_cached_tokens,
                    vllm_cached_tokens=vllm_cached_tokens,
                    all_token_ids=all_token_ids,
                )
                sync_authoritative_block_ids(
                    request_tracker, req.req_id, "cached_request"
                )

                old_saved_tokens = apply_specagent_scratch_save_floor(
                    request_tracker,
                    req.req_id,
                )
                old_skip_save = set_step_skip_save(request_tracker, req.req_id)
                req_meta = ReqMeta.from_request_tracker(
                    request_tracker,
                    self._block_size,
                    self._lmcache_chunk_size,
                    load_spec=load_spec,
                    discard_partial_chunks=self._discard_partial_chunks,
                    save_decode_cache=self.config.save_decode_cache,
                    force_save=specagent_step_force_save(req.req_id),
                )
                debug_specagent_req_meta(
                    "cached_list/from_tracker",
                    req.req_id,
                    request_tracker,
                    req_meta,
                    load_spec,
                )
                restore_step_skip_save(request_tracker, old_skip_save)
                restore_specagent_scratch_save_floor_if_unsaved(
                    request_tracker,
                    req.req_id,
                    old_saved_tokens,
                    req_meta,
                )
                req_meta = sanitize_specagent_scratch_save_meta(
                    req.req_id,
                    req_meta,
                )
                debug_specagent_req_meta(
                    "cached_list/sanitized",
                    req.req_id,
                    request_tracker,
                    req_meta,
                    load_spec,
                )
                if req_meta is not None:
                    meta.add_request(req_meta)
                    emitted_req_ids.add(req.req_id)
            emit_missing_specagent_fetch_meta()
            return meta

        _tl_t0 = time.perf_counter()
        for i, req_id in enumerate(cached_reqs.req_ids):
            request_tracker = self._request_trackers.get(req_id)
            if request_tracker is None:
                # (2026-08-18) attach/DRAFT 경로로 첫 스케줄이 cached 로
                # 분류된 요청 — tracker 미등록. 엔진을 죽이는 대신 이번
                # 스텝 메타를 생략(그 턴은 커넥터 저장/로드 없이 진행,
                # 재계산 경로가 정확성 보장). 배선 근본수정용 단서 로그.
                logger.warning(
                    "[SpecAgent][TRACKER-MISS] cached req without tracker "
                    "req=%s -- skipping connector meta this step", req_id)
                continue
            num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            # TODO: this is a dangerous reference to the request object inside vllm
            if request := self._unfinished_requests.get(req_id):
                num_current_tokens = request.num_computed_tokens
                # tracker_len < num_computed_tokens during decode
                #   (important for save_decode_cache).
                # num_computed_tokens < tracker_len after preemption.
                tracker_len = len(request_tracker.token_ids)
                slice_base = min(num_current_tokens, tracker_len)
                new_token_ids = request.all_token_ids[
                    slice_base : slice_base + num_new_tokens
                ]
            else:
                raise ValueError(
                    f"Request {req_id} is not in _unfinished_requests, "
                    f"but it is scheduled to be cached"
                )
            new_block_ids = cached_reqs.new_block_ids[i]

            load_spec = self.load_specs.pop(req_id, None)
            if req_id in specagent_fetch_reqs:
                forced_load_spec = get_specagent_load_spec(req_id)
                if forced_load_spec is not None:
                    load_spec = forced_load_spec
            elif load_spec is None:
                load_spec = get_specagent_load_spec(req_id)
            lmcache_cached_tokens = 0
            vllm_cached_tokens = 0
            if load_spec is not None:
                lmcache_cached_tokens = load_spec.lmcache_cached_tokens
                vllm_cached_tokens = load_spec.vllm_cached_tokens

            # Handle both old and new versions of CachedRequestData
            if hasattr(cached_reqs, "resumed_req_ids"):
                # New version with resumed_req_ids
                preempted = req_id in cached_reqs.resumed_req_ids
            elif hasattr(cached_reqs, "resumed_from_preemption"):
                # Old version with resumed_from_preemption
                preempted = cached_reqs.resumed_from_preemption[i]
            else:
                # This case should not be reached with supported vLLM versions.
                # Raising an error is safer than assuming not preempted.
                raise AttributeError(
                    f"Unable to determine preemption status for request {req_id}. "
                    f"This might be due to an unsupported vLLM version."
                )
            if preempted:
                if load_spec is None:
                    # A resumed request may have only vLLM-local prefix cache
                    # available.  Do not invent an LMCache hit here: the
                    # scheduler did not reserve external-load slots for one,
                    # and marking tokens as LMCache-backed would violate the
                    # SpecAgent CPU-backed proof invariant.
                    vllm_cached_tokens = max(0, int(request.num_computed_tokens))
                    lmcache_cached_tokens = 0
                    load_spec = LoadSpec(
                        vllm_cached_tokens=vllm_cached_tokens,
                        lmcache_cached_tokens=lmcache_cached_tokens,
                        can_load=False,
                    )
                    logger.warning(
                        "[SpecAgent][PREEMPT-NO-LOADSPEC] req=%s "
                        "num_computed_tokens=%d num_external_computed_tokens=%s "
                        "num_new_tokens=%d tracker_len=%d; using vLLM-local "
                        "no-load fallback",
                        req_id,
                        request.num_computed_tokens,
                        getattr(request, "num_external_computed_tokens", None),
                        num_new_tokens,
                        len(request_tracker.token_ids),
                    )
                # num_computed_tokens should be reset to 0 during preemption
                # and then set to the number of already cached tokens (maxxing
                # prefix caching and lmcache)
                # this assumption is crucial for the update() call of RequestTracker
                # On full cache hit, get_num_new_matched_tokens subtracts 1
                # to force last-token recomputation. This only affects
                # num_computed_tokens when lmcache has all tokens AND
                # provides more than vLLM's local cache.
                expected = max(lmcache_cached_tokens, load_spec.vllm_cached_tokens)
                full_hit_adj = (
                    lmcache_cached_tokens == len(request.all_token_ids)
                    and lmcache_cached_tokens > load_spec.vllm_cached_tokens
                )
                if full_hit_adj:
                    expected -= 1
                if request.num_computed_tokens != expected:
                    logger.warning(
                        "[SpecAgent][PREEMPT-LOAD-MISMATCH] req=%s "
                        "num_computed_tokens=%d expected=%d "
                        "lmcache_cached_tokens=%d vllm_cached_tokens=%d "
                        "full_hit_adj=%s; capping load_spec to actual "
                        "computed prefix to avoid saving beyond valid KV.",
                        req_id,
                        request.num_computed_tokens,
                        expected,
                        lmcache_cached_tokens,
                        load_spec.vllm_cached_tokens,
                        full_hit_adj,
                    )
                    lmcache_cached_tokens = min(
                        lmcache_cached_tokens, request.num_computed_tokens,
                    )
                    vllm_cached_tokens = min(
                        vllm_cached_tokens, request.num_computed_tokens,
                    )
                    load_spec = LoadSpec(
                        vllm_cached_tokens=vllm_cached_tokens,
                        lmcache_cached_tokens=lmcache_cached_tokens,
                        can_load=load_spec.can_load,
                    )

            # When retrieve fail, vllm will call _handle_invalid_blocks to
            # reset request.num_computed_tokens, this will lead to
            # request_tracker.token_ids being not matched with vllm
            if num_current_tokens < len(request_tracker.token_ids):
                logger.warning(
                    "Request %s rolled back from %d to %d tokens; "
                    "truncating tracker state.",
                    req_id,
                    len(request_tracker.token_ids),
                    num_current_tokens,
                )
                num_token_slots = (
                    len(request_tracker.allocated_block_ids) * self._block_size
                )
                tokens_to_keep = num_current_tokens
                if num_token_slots < num_current_tokens:
                    logger.warning(
                        "Request %s tracker has %d token slots but %d tokens; "
                        "capping token_ids to slot capacity.",
                        req_id,
                        num_token_slots,
                        num_current_tokens,
                    )
                    tokens_to_keep = num_token_slots

                request_tracker.token_ids = list(request.all_token_ids[:tokens_to_keep])
                request_tracker.num_saved_tokens = min(
                    request_tracker.num_saved_tokens, tokens_to_keep
                )

            # Pass all_token_ids for preempted requests to restore
            # token_ids correctly for chunk key computation
            all_token_ids = list(request.all_token_ids) if preempted else None

            tracker_lmcache_cached_tokens = (
                request_tracker.num_saved_tokens
                if req_id in specagent_fetch_reqs
                else lmcache_cached_tokens
            )
            if req_id in specagent_fetch_reqs and lmcache_cached_tokens:
                logger.warning(
                    "[SpecAgent][SAVE-META] req=%s forced_load_tokens=%d "
                    "tracker_saved_tokens_kept=%d",
                    req_id,
                    lmcache_cached_tokens,
                    tracker_lmcache_cached_tokens,
                )

            request_tracker.update(
                new_token_ids,
                new_block_ids,
                preempted=preempted,
                lmcache_cached_tokens=tracker_lmcache_cached_tokens,
                vllm_cached_tokens=vllm_cached_tokens,
                all_token_ids=all_token_ids,
            )
            sync_authoritative_block_ids(
                request_tracker, req_id, "cached_request"
            )

            old_saved_tokens = apply_specagent_scratch_save_floor(
                request_tracker,
                req_id,
            )
            old_skip_save = set_step_skip_save(request_tracker, req_id)
            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                self._lmcache_chunk_size,
                load_spec=load_spec,
                discard_partial_chunks=self._discard_partial_chunks,
                save_decode_cache=self.config.save_decode_cache,
                force_save=specagent_step_force_save(req_id),
            )
            debug_specagent_req_meta(
                "cached_request/from_tracker",
                req_id,
                request_tracker,
                req_meta,
                load_spec,
            )
            restore_step_skip_save(request_tracker, old_skip_save)
            restore_specagent_scratch_save_floor_if_unsaved(
                request_tracker,
                req_id,
                old_saved_tokens,
                req_meta,
            )
            req_meta = sanitize_specagent_scratch_save_meta(req_id, req_meta)
            debug_specagent_req_meta(
                "cached_request/sanitized",
                req_id,
                request_tracker,
                req_meta,
                load_spec,
            )
            if req_meta is not None:
                meta.add_request(req_meta)
                emitted_req_ids.add(req_id)

        builtins._lmc_ms_tracker_loop = getattr(
            builtins, "_lmc_ms_tracker_loop", 0.0) + (
            time.perf_counter() - _tl_t0) * 1000.0
        emit_missing_specagent_fetch_meta()
        return meta

    @_lmcache_nvtx_annotate
    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        # Layerwise save uses request-scoped generators. If request finishes
        # without entering wait_for_save (abort/error/evict path), make sure
        # we release the generator entry to avoid leaking state.
        if getattr(self, "use_layerwise", False) and hasattr(
            self, "_layerwise_save_storers"
        ):
            self._specagent_join_save_futures()
            self._layerwise_save_storers.pop(request.request_id, None)

        # Cleanup if request was aborted
        if request.status == RequestStatus.FINISHED_ABORTED and self.async_loading:
            # Cancel any ongoing async lookup and prefetch tasks on workers
            lookup_id = request.request_id
            assert self.lookup_client is not None
            self.lookup_client.cancel_lookup(lookup_id)  # type: ignore[attr-defined]

        params = (
            request.kv_transfer_params
            if hasattr(request, "kv_transfer_params")
            else None
        )
        return_params = None

        # NOTE: Used to stream back the first token
        # for disagg prefill
        if params is not None and "ret_first_tok" in params:
            return_params = {
                "first_tok": request._output_token_ids[0],
            }

        if self.config.get_extra_config_value(
            "enable_cache_usage_details_in_response", False
        ):
            request_tracker = self._request_trackers.get(request.request_id)
            if request_tracker:
                return_params = return_params or {}
                return_params["num_lmcache_cached_tokens"] = (
                    request_tracker.num_lmcache_cached_tokens
                )

        spec_cfg = getattr(self._vllm_config, "specagent_config", None)
        if isinstance(spec_cfg, dict):
            turn_end_only = bool(spec_cfg.get("save_kv_on_turn_end_only", False))
        else:
            turn_end_only = bool(
                getattr(spec_cfg, "save_kv_on_turn_end_only", False)
                if spec_cfg is not None
                else False
            )
        if (
            turn_end_only
            and getattr(request, "job_id", None) is not None
            and request.status != RequestStatus.FINISHED_ABORTED
        ):
            tracker = self._request_trackers.get(request.request_id)
            skip_leading_tokens = (
                int(tracker.num_saved_tokens)
                if tracker is not None else 0
            )
            # The final sampled output token has not been used as a model
            # input yet, so its paged slot does not contain valid KV even when
            # async scheduler accounting reports num_computed == num_tokens.
            # Publishing it advances the saved-token tracker over a zero row;
            # a later sparse/paged restore then faithfully restores corrupted
            # KV. The next turn will compute this token and a later save can
            # safely include it.
            valid_kv_token_limit = max(
                0,
                len(request.all_token_ids)
                - (1 if request.output_token_ids else 0),
            )
            token_limit = min(
                int(getattr(request, "num_computed_tokens", 0) or 0),
                valid_kv_token_limit,
                len(request.all_token_ids),
                len(block_ids) * self._block_size,
            )
            if token_limit > skip_leading_tokens:
                block_ids_tensor = torch.tensor(block_ids, dtype=torch.long)
                block_offsets = torch.arange(
                    0, self._block_size, dtype=torch.long
                )
                slot_mapping = (
                    block_offsets.reshape((1, self._block_size))
                    + block_ids_tensor.reshape((len(block_ids), 1))
                    * self._block_size
                ).flatten()[:token_limit]
                req_meta = ReqMeta(
                    req_id=request.request_id,
                    token_ids=list(request.all_token_ids[:token_limit]),
                    slot_mapping=slot_mapping,
                    is_last_prefill=True,
                    save_spec=SaveSpec(skip_leading_tokens, True),
                    load_spec=None,
                    disagg_spec=None,
                    request_configs=(
                        tracker.request_configs
                        if tracker is not None else None
                    ),
                    is_finished_save=True,
                )
                self._specagent_finished_save_metas[request.request_id] = req_meta
                builtins._sa_finmeta_emit = getattr(
                    builtins, "_sa_finmeta_emit", 0) + 1
                if _specagent_env_true("SPECAGENT_SAVE_DEBUG"):
                    logger.warning(
                        "[SpecAgent][FINISH-SAVE] req=%s tokens=%d "
                        "skip=%d blocks=%d status=%s",
                        request.request_id,
                        token_limit,
                        skip_leading_tokens,
                        len(block_ids),
                        request.status,
                    )
                return True, return_params

        return False, return_params

    def specagent_register_preempt_save(
        self,
        request: "Request",
        block_ids: list[int],
        save_upto_tokens: int,
        save_id: int,
    ) -> int:
        """Register a mid-turn CPU save of ``request``'s confirmed KV region.

        Scheduler-side.  Mirrors the turn-end finished-save meta construction in
        ``request_finished`` but (1) bounds the save to ``save_upto_tokens`` (the
        VERIFIED/confirmed boundary, so unverified draft KV is never written) and
        (2) marks the meta ``is_preempt_save`` so the worker stores it WITHOUT
        reporting the request as finished_sending (the request is still alive).

        Returns the token boundary requested from this save/probe. The worker
        ACK separately reports the prefix LMCache can actually resolve after
        the store lands. Returns 0 if no complete chunk can be covered.
        """
        if self.kv_role == "kv_consumer":
            return 0
        if getattr(request, "job_id", None) is None:
            return 0
        tracker = self._request_trackers.get(request.request_id)
        skip_leading_tokens = (
            int(tracker.num_saved_tokens) if tracker is not None else 0
        )
        valid_kv_token_limit = max(
            0,
            len(request.all_token_ids)
            - (1 if request.output_token_ids else 0),
        )
        token_limit = min(
            int(save_upto_tokens),
            valid_kv_token_limit,
            len(request.all_token_ids),
            len(block_ids) * self._block_size,
        )
        # Align DOWN to the LMCache chunk boundary.  An unaligned save
        # advances tracker.num_saved_tokens past the last complete chunk,
        # so the later turn-end save skips the [aligned, unaligned) piece
        # and the containing chunk is never completed — the coverage
        # ledger then caps at the previous chunk forever, and the next
        # turn crashes with FETCH-UNRECOVERABLE once the GPU ghosts of
        # that region are reallocated.  The dropped partial-chunk tail is
        # recomputed on resume, which the preempt path already assumes.
        chunk_size = max(1, int(self._lmcache_chunk_size or 256))
        token_limit = (token_limit // chunk_size) * chunk_size
        if token_limit <= 0:
            return 0
        if token_limit <= skip_leading_tokens:
            # Emit a zero-copy coverage probe. A stale tracker is not proof:
            # the worker ACK reports the prefix LMCache can actually find.
            skip_leading_tokens = token_limit
        block_ids_tensor = torch.tensor(block_ids, dtype=torch.long)
        block_offsets = torch.arange(0, self._block_size, dtype=torch.long)
        slot_mapping = (
            block_offsets.reshape((1, self._block_size))
            + block_ids_tensor.reshape((len(block_ids), 1)) * self._block_size
        ).flatten()[:token_limit]
        req_meta = ReqMeta(
            req_id=request.request_id,
            token_ids=list(request.all_token_ids[:token_limit]),
            slot_mapping=slot_mapping,
            is_last_prefill=True,
            save_spec=SaveSpec(skip_leading_tokens, True),
            load_spec=None,
            disagg_spec=None,
            request_configs=(
                tracker.request_configs if tracker is not None else None
            ),
            is_finished_save=False,
            is_preempt_save=True,
            specagent_preempt_save_id=int(save_id),
        )
        self._specagent_preempt_save_metas[request.request_id] = req_meta
        if _specagent_env_true("SPECAGENT_SAVE_DEBUG"):
            logger.warning(
                "[SpecAgent][PREEMPT-SAVE] register req=%s tokens=%d skip=%d "
                "blocks=%d",
                request.request_id, token_limit, skip_leading_tokens,
                len(block_ids),
            )
        return token_limit

    @_lmcache_nvtx_annotate
    def get_kv_events(self) -> Iterable[CacheStoreEvent]:
        if self.lmcache_engine is not None:
            return self.lmcache_engine.get_kv_events()
        return []
