# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional, Tuple, Union
import abc
import builtins
import os
import threading
import time
from contextlib import nullcontext

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import EngineType, _lmcache_nvtx_annotate
from lmcache.v1.compute.blend.utils import LMCBlenderBuilder
from lmcache.v1.gpu_connector.utils import (
    LayoutHints,
    assert_is_vllm_flash_attn_or_flash_infer,
    discover_gpu_kv_format,
    ensure_contiguous_kv_caches,
    get_block_size,
    get_elements_per_layer,
    get_head_size,
    get_num_blocks,
    get_page_buffer_size,
    get_tokens_per_layer,
    is_hnd,
    permute_kv_caches_to_contiguous,
)
from lmcache.v1.memory_management import GPUMemoryAllocator  # noqa: E501
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata

if torch.cuda.is_available():
    # First Party
    import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)


def _specagent_record_cuda_event(
    stream: torch.cuda.Stream,
    *,
    enable_timing: bool = False,
) -> torch.cuda.Event:
    lock = getattr(builtins, "_specagent_cuda_event_record_lock", None)
    if lock is None:
        lock = threading.Lock()
        builtins._specagent_cuda_event_record_lock = lock
    with lock:
        event = torch.cuda.Event(enable_timing=enable_timing)
        event.record(stream)
    return event


def _next_power_of_2(value: int) -> int:
    return 1 << max(0, (value - 1).bit_length())


def _can_raw_h2d_copy(dst: torch.Tensor, src: torch.Tensor) -> bool:
    return (
        dst.device.type == "cuda"
        and src.device.type == "cpu"
        and dst.is_contiguous()
        and src.is_contiguous()
        and dst.numel() == src.numel()
        and dst.element_size() == src.element_size()
    )


def _raw_h2d_copy_async(dst: torch.Tensor, src: torch.Tensor) -> None:
    nbytes = int(src.numel() * src.element_size())
    if nbytes == 0:
        return
    # LMCache CPU buffers are cudaHostAlloc/cudaHostRegister-backed but are
    # exposed to PyTorch via torch.frombuffer(), so Tensor.is_pinned() is false.
    # Use LMCache's raw-pointer memcpyAsync path to avoid PyTorch staging.
    lmc_ops.lmcache_memcpy_async(
        dst.data_ptr(),
        src.data_ptr(),
        nbytes,
        lmc_ops.TransferDirection.H2D,
        0,
        _next_power_of_2(nbytes),
    )


def _raw_h2d_copy_batch_async(
    dsts: list[torch.Tensor],
    srcs: list[torch.Tensor],
) -> None:
    if not dsts:
        return
    # CUDA 12.8 can submit independent scatter/gather copies as one batch.
    # Keep the older per-range implementation as an explicit compatibility
    # fallback.  Deferred plans retain their pinned sources and temporary GPU
    # allocation until the recorded ready event completes, so neither side of
    # the asynchronous batch can be recycled while CUDA still consumes it.
    native_batch_copy = getattr(lmc_ops, "lmcache_memcpy_batch_async", None)
    if (
        native_batch_copy is not None
        and not os.environ.get("SPECAGENT_DISABLE_CUDA_MEMCPY_BATCH")
    ):
        native_batch_copy(
            [int(dst.data_ptr()) for dst in dsts],
            [int(src.data_ptr()) for src in srcs],
            [int(src.numel() * src.element_size()) for src in srcs],
            lmc_ops.TransferDirection.H2D,
        )
        return

    batch_copy = getattr(lmc_ops, "lmcache_memcpy_async_batch", None)
    if batch_copy is None:
        for dst, src in zip(dsts, srcs, strict=False):
            _raw_h2d_copy_async(dst, src)
        return

    nbytes = [int(src.numel() * src.element_size()) for src in srcs]
    batch_copy(
        [int(dst.data_ptr()) for dst in dsts],
        [int(src.data_ptr()) for src in srcs],
        nbytes,
        lmc_ops.TransferDirection.H2D,
        [0] * len(nbytes),
        [_next_power_of_2(num_bytes) for num_bytes in nbytes],
    )


try:
    # vLLM emits these ranges into Nsight when profiling is enabled.  Keep
    # LMCache importable in standalone contexts where vLLM is absent.
    from vllm.v1.utils import record_function_or_nullcontext as _lmc_record
except Exception:  # pragma: no cover - optional vLLM dependency.
    def _lmc_record(_name: str):
        return nullcontext()


class GPUConnectorInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        # FIXME (Yihua): We shouldn't put start and end here since
        # it's not the responsibility of the GPUConnector to know
        # the token-sequence-related information.
        """Store the data in the memory object into a GPU buffer.
        Sub-classes should define the format of the kwargs.

        :param MemoryObj memory_obj: The memory object to be copied into GPU.
        :param int start: The starting index of the data in the corresponding
            token sequence.
        :param int end: The ending index of the data in the corresponding
            token sequence.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        # FIXME (Yihua): We shouldn't put start and end here since
        # it's not the responsibility of the GPUConnector to know
        # the token-sequence-related information.
        """Load the data from a GPU buffer into the memory object.
        Sub-classes should define the format of the kwargs.

        :param MemoryObj memory_obj: The memory object to store the data from
            GPU.
        :param int start: The starting index of the data in the corresponding
            token sequence.
        :param int end: The ending index of the data in the corresponding
            token sequence.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]], List[MemoryObj]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """
        Batched load the data from a GPU memory into the memory objects.
        Sub-classes should define the format of the kwargs.

        :param Union[List[List[MemoryObj]], List[MemoryObj]] memory_obj:
            The memory objects to store the data from GPU.
        :param List[int] starts: The starting indices of the data in the corresponding
            token sequence.
        :param List[int] ends: The ending indices of the data in the corresponding
            token sequence.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def batched_to_gpu(
        self,
        memory_objs: Union[
            List[List[MemoryObj]], List[MemoryObj], List[int], None
        ] = None,
        starts: Optional[List[int]] = None,
        ends: Optional[List[int]] = None,
        **kwargs,
    ):
        """
        Batched store the data from the memory objects to GPU kv cache.
        Sub-classes should define the format of the kwargs.

        For non-layerwise connectors:
        :param Union[List[List[MemoryObj]], List[MemoryObj]] memory_obj:
            The memory objects to store the data to GPU.
        :param List[int] starts: The starting indices of the data in the corresponding
            token sequence.
        :param List[int] ends: The ending indices of the data in the corresponding
            token sequence.

        For layerwise connectors (generator pattern):
        :param List[int] memory_objs: Actually the starts list
        (positional compatibility)
        :param List[int] starts: Actually the ends list
        (positional compatibility)
        Note: Layerwise connectors receive memory objects
        via generator.send()
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_shape(self, num_tokens: int) -> torch.Size:
        """Get the shape of the data given the number of tokens."""
        raise NotImplementedError

    def initialize_kvcaches_ptr(self, **kwargs):
        """Initialize the kvcaches pointers if not already initialized."""
        if "kvcaches" in kwargs:
            self.kvcaches = kwargs["kvcaches"]
            # Ensure contiguity on every call.  HND tensors from vLLM have a
            # non-contiguous logical view (NHD) that must be permuted back to
            # the physical (HND) shape for correct kernel indexing.
            # permute_kv_caches_to_contiguous is a no-op when already contiguous.
            self.kvcaches = permute_kv_caches_to_contiguous(self.kvcaches)


class VLLMPagedMemGPUConnectorV2(GPUConnectorInterface):
    """
    The GPU KV cache should be a nested tuple of K and V tensors.
    More specifically, we have:
    - GPUTensor = Tuple[KVLayer, ...]
    - KVLayer = Tuple[Tensor, Tensor]
    - Tensor: [num_blocks, block_size, num_heads, head_size]

    It will produce / consume memory object with KV_2LTD format
    """

    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        **kwargs,
    ):
        """
        If use_gpu is true, it will create a gpu intermediate buffer. In this
        case, it requires the following kwargs:
        - chunk_size: The MAX size of the chunk to be copied to GPU.
        - dtype: The data type of the intermediate buffer.
        """
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.kv_cache_pointers = torch.empty(
            num_layers, dtype=torch.int64, device="cpu"
        )
        # Not sure we need a dict here. Maybe a single GPU connector always
        # works with a single device?
        self.kv_cache_pointers_on_gpu: dict[int, torch.Tensor] = {}

        self.kvcaches: Optional[List[torch.Tensor]] = None

        self.gpu_buffer: Optional[torch.Tensor] = None
        self.use_mla = "use_mla" in kwargs and kwargs["use_mla"]
        self.layout_hints: LayoutHints = (
            kwargs.get(  # type: ignore[assignment]
                "layout_hints"
            )
            or {}
        )
        if use_gpu:
            assert "chunk_size" in kwargs, (
                "chunk_size should be provided to create a GPU buffer."
            )
            assert "dtype" in kwargs, "dtype should be provided to create a GPU buffer."
            assert "device" in kwargs, (
                "device should be provided to create a GPU buffer."
            )
            shape = self.get_shape(kwargs["chunk_size"])
            self.gpu_buffer = torch.empty(
                shape, dtype=kwargs["dtype"], device=kwargs["device"]
            )

        self.store_stream = torch.cuda.Stream()
        self.load_stream = torch.cuda.Stream()

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheMetadata,
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
        layout_hints: Optional[LayoutHints] = None,
    ) -> "VLLMPagedMemGPUConnectorV2":
        """Create a connector from LMCacheMetadata.

        Args:
            metadata: The LMCache engine metadata containing model configuration.
            use_gpu: Whether to use GPU intermediate buffer.
            device: The device to use for the connector.
            layout_hints: Optional hints about KV cache layout from the
                serving engine.

        Returns:
            A new instance of VLLMPagedMemGPUConnectorV2.
        """
        # Extract parameters from metadata
        # kv_shape: (num_layer, 2 or 1, chunk_size, num_kv_head, head_size)
        num_layers = metadata.kv_shape[0]
        chunk_size = metadata.kv_shape[2]
        num_kv_head = metadata.kv_shape[3]
        head_size = metadata.kv_shape[4]
        hidden_dim_size = num_kv_head * head_size

        return cls(
            hidden_dim_size=hidden_dim_size,
            num_layers=num_layers,
            use_gpu=use_gpu,
            chunk_size=chunk_size,
            dtype=metadata.kv_dtype,
            device=device,
            use_mla=metadata.use_mla,
            layout_hints=layout_hints,
        )

    def _initialize_pointers(self, kv_caches: List[torch.Tensor]) -> torch.Tensor:
        self.device = kv_caches[0].device
        assert self.device.type == "cuda", "The device should be CUDA."
        idx = self.device.index
        if idx in self.kv_cache_pointers_on_gpu:
            return self.kv_cache_pointers_on_gpu[idx]

        # contiguous before pointer capture or format discovery
        kv_caches = ensure_contiguous_kv_caches(
            kv_caches, kv_layout=self.layout_hints.get("kv_layout")
        )

        self.kv_cache_pointers.numpy()[:] = [t.data_ptr() for t in kv_caches]
        self.kv_cache_pointers_on_gpu[idx] = torch.empty(
            self.num_layers, dtype=torch.int64, device=self.device
        )
        self.kv_cache_pointers_on_gpu[idx].copy_(self.kv_cache_pointers)

        self.gpu_kv_format = discover_gpu_kv_format(
            kv_caches, EngineType.VLLM, layout_hints=self.layout_hints
        )
        self.num_blocks = get_num_blocks(kv_caches, self.gpu_kv_format)
        self.block_size = get_block_size(kv_caches, self.gpu_kv_format)
        self.page_buffer_size = self.num_blocks * self.block_size
        self.head_size = get_head_size(kv_caches, self.gpu_kv_format)

        return self.kv_cache_pointers_on_gpu[idx]

    @_lmcache_nvtx_annotate
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Note:
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)


        :raises ValueError: If 'kvcaches' is not provided in kwargs.
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        self.initialize_kvcaches_ptr(**kwargs)

        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if self.use_mla:
            if memory_obj.metadata.fmt != MemoryFormat.KV_MLA_FMT:
                raise ValueError(
                    "The memory object should be in KV_MLA_FMT format in"
                    " order to be processed by VLLMPagedMemGPUConnector"
                )
        else:
            if memory_obj.metadata.fmt != MemoryFormat.KV_2LTD:
                raise ValueError(
                    "The memory object should be in KV_2LTD format in"
                    " order to be processed by VLLMPagedMemGPUConnector"
                )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        kv_cache_pointers = self._initialize_pointers(self.kvcaches)

        # avoid read/write stream race condition for shared block
        # this will only be potentially non-zero for the first
        # block lmcache is transferring back
        vllm_cached = kwargs.get("vllm_cached_tokens", 0)
        skip_prefix_n_tokens = min(end - start, max(0, vllm_cached - start))

        lmc_ops.multi_layer_kv_transfer(
            memory_obj.tensor,
            kv_cache_pointers,
            slot_mapping[start:end],
            self.device,
            self.page_buffer_size,
            lmc_ops.TransferDirection.H2D,
            self.gpu_kv_format,
            block_size=self.block_size,
            head_size=self.head_size,
            skip_prefix_n_tokens=skip_prefix_n_tokens,
        )

    @_lmcache_nvtx_annotate
    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Will set the memory_obj.metadata.fmt to MemoryFormat.KV_2LTD.

        Note:
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)

        :raises ValueError: If 'kvcaches' is not provided in kwargs,
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )
        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        kv_cache_pointers = self._initialize_pointers(self.kvcaches)

        with torch.cuda.stream(self.store_stream):
            if self.gpu_buffer is None or end - start != self.gpu_buffer.shape[2]:
                lmc_ops.multi_layer_kv_transfer(
                    memory_obj.tensor,
                    kv_cache_pointers,
                    slot_mapping[start:end],
                    self.kvcaches[0].device,
                    self.page_buffer_size,
                    lmc_ops.TransferDirection.D2H,
                    self.gpu_kv_format,
                    block_size=self.block_size,
                    head_size=self.head_size,
                )
            else:
                # kvcaches -> gpu_buffer -> memobj
                assert self.gpu_buffer.device == self.kvcaches[0].device
                tmp_gpu_buffer = self.gpu_buffer[:, :, : end - start, :]
                lmc_ops.multi_layer_kv_transfer(
                    tmp_gpu_buffer,
                    kv_cache_pointers,
                    slot_mapping[start:end],
                    self.kvcaches[0].device,
                    self.page_buffer_size,
                    lmc_ops.TransferDirection.D2H,
                    self.gpu_kv_format,
                    block_size=self.block_size,
                    head_size=self.head_size,
                )
                memory_obj.tensor.copy_(tmp_gpu_buffer, non_blocking=True)

        if not memory_obj.tensor.is_cuda:
            # Force a synchronize if the target buffer is NOT CUDA device
            # NOTE: for better performance, we may not want to sync for every
            # memory object
            self.store_stream.synchronize()

        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

    # TODO(Jiayi): need to optimize to enable real batching
    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        with torch.cuda.stream(self.load_stream):
            for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
                self.to_gpu(memory_obj, start, end, **kwargs)
        self.load_stream.synchronize()

    # TODO(Jiayi): need to optimize to enable real batching
    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.from_gpu(memory_obj, start, end, **kwargs)

    def get_shape(self, num_tokens: int) -> torch.Size:
        kv_size = 1 if self.use_mla else 2
        return torch.Size([kv_size, self.num_layers, num_tokens, self.hidden_dim_size])


class VLLMPagedMemGPUConnectorV3(GPUConnectorInterface):
    def __init__(
        self,
        metadata: LMCacheMetadata,
        device: torch.device,
        use_gpu: bool = False,
        layout_hints: Optional[LayoutHints] = None,
    ):
        assert device.type == "cuda", "The device should be CUDA."
        self.metadata = metadata
        self.device = device
        self.use_mla = metadata.use_mla
        self.chunk_size = metadata.chunk_size
        self.use_gpu = use_gpu
        self.layout_hints: LayoutHints = layout_hints or {}
        self.kvcaches: Optional[List[torch.Tensor]] = None

        self.init = False
        self.group_kv_cache_pointers_on_gpu: Optional[list[torch.Tensor]] = None
        self.group_tmp_buffer: Optional[list[torch.Tensor]] = None

        self.store_stream = torch.cuda.Stream()
        self.load_stream = torch.cuda.Stream()

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheMetadata,
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
        layout_hints: Optional[LayoutHints] = None,
    ) -> "VLLMPagedMemGPUConnectorV3":
        assert device is not None
        return cls(metadata, device, use_gpu, layout_hints=layout_hints)

    def _initialize_kv_cache_pointers(self):
        if self.init:
            return
        assert self.metadata.kv_layer_groups_manager.kv_layer_groups

        # permute to contiguous before capturing pointers or doing format discovery
        self.kvcaches = ensure_contiguous_kv_caches(
            self.kvcaches, kv_layout=self.layout_hints.get("kv_layout")
        )

        if self.use_gpu:
            # init tmp buffer
            tmp_buf_shapes = self.metadata.get_shapes(self.chunk_size)
            tmp_buf_dtypes = self.metadata.get_dtypes()
            assert len(tmp_buf_shapes) == len(tmp_buf_dtypes)
            self.group_tmp_buffer = [
                torch.empty(tmp_buf_shape, dtype=tmp_buf_dtype, device=self.device)
                for tmp_buf_shape, tmp_buf_dtype in zip(
                    tmp_buf_shapes, tmp_buf_dtypes, strict=True
                )
            ]
        self.group_kv_cache_pointers_on_gpu = []
        for group in self.metadata.kv_layer_groups_manager.kv_layer_groups:
            # init kv cache pointers
            num_layers = group.num_layers
            kv_cache_pointers = torch.empty(num_layers, dtype=torch.int64, device="cpu")
            kv_cache_pointers.numpy()[:] = [
                t.data_ptr()
                for i, t in enumerate(self.kvcaches)
                if i in group.layer_indices
            ]
            kv_cache_pointers_on_gpu = torch.empty(
                num_layers, dtype=torch.int64, device=self.device
            )
            kv_cache_pointers_on_gpu.copy_(kv_cache_pointers)
            self.group_kv_cache_pointers_on_gpu.append(kv_cache_pointers_on_gpu)

        self.gpu_kv_format = discover_gpu_kv_format(
            self.kvcaches, EngineType.VLLM, layout_hints=self.layout_hints
        )
        self.num_blocks = get_num_blocks(self.kvcaches, self.gpu_kv_format)
        self.block_size = get_block_size(self.kvcaches, self.gpu_kv_format)
        self.page_buffer_size = self.num_blocks * self.block_size
        self.head_size = get_head_size(self.kvcaches, self.gpu_kv_format)

        self.init = True
        logger.info("init kv cache pointers success in VLLMPagedMemGPUConnectorV3")

    @_lmcache_nvtx_annotate
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        assert memory_obj.raw_tensor is not None
        assert "slot_mapping" in kwargs
        if self.use_mla:
            assert memory_obj.metadata.fmt == MemoryFormat.KV_MLA_FMT
        else:
            assert memory_obj.metadata.fmt == MemoryFormat.KV_2LTD

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None
        assert self.kvcaches[0].device == self.device
        self._initialize_kv_cache_pointers()
        assert self.group_kv_cache_pointers_on_gpu is not None

        # avoid read/write stream race condition for shared block
        # this will only be potentially non-zero for the first
        # block lmcache is transferring back
        vllm_cached = kwargs.get("vllm_cached_tokens", 0)
        skip_prefix_n_tokens = min(end - start, max(0, vllm_cached - start))

        for i, kv_cache_pointer in enumerate(self.group_kv_cache_pointers_on_gpu):
            memory_obj_tensor = memory_obj.get_tensor(i)
            assert memory_obj_tensor is not None
            lmc_ops.multi_layer_kv_transfer(
                memory_obj_tensor,
                kv_cache_pointer,
                slot_mapping[start:end],
                self.device,
                self.page_buffer_size,
                lmc_ops.TransferDirection.H2D,
                self.gpu_kv_format,
                block_size=self.block_size,
                head_size=self.head_size,
                skip_prefix_n_tokens=skip_prefix_n_tokens,
            )

    @_lmcache_nvtx_annotate
    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        assert memory_obj.raw_tensor is not None
        assert "slot_mapping" in kwargs

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None
        assert self.kvcaches[0].device == self.device
        self._initialize_kv_cache_pointers()
        assert self.group_kv_cache_pointers_on_gpu is not None
        with torch.cuda.stream(self.store_stream):
            if not self.use_gpu or end - start != self.chunk_size:
                for i, kv_cache_pointer in enumerate(
                    self.group_kv_cache_pointers_on_gpu
                ):
                    memory_obj_tensor = memory_obj.get_tensor(i)
                    assert memory_obj_tensor is not None
                    lmc_ops.multi_layer_kv_transfer(
                        memory_obj_tensor,
                        kv_cache_pointer,
                        slot_mapping[start:end],
                        self.device,
                        self.page_buffer_size,
                        lmc_ops.TransferDirection.D2H,
                        self.gpu_kv_format,
                        block_size=self.block_size,
                        head_size=self.head_size,
                    )
            else:
                # kvcaches -> gpu_buffer -> memobj
                assert self.group_tmp_buffer is not None
                for i, kv_cache_pointer in enumerate(
                    self.group_kv_cache_pointers_on_gpu
                ):
                    tmp_gpu_buffer = self.group_tmp_buffer[i][:, :, : end - start, :]
                    lmc_ops.multi_layer_kv_transfer(
                        tmp_gpu_buffer,
                        kv_cache_pointer,
                        slot_mapping[start:end],
                        self.device,
                        self.page_buffer_size,
                        lmc_ops.TransferDirection.D2H,
                        self.gpu_kv_format,
                        block_size=self.block_size,
                        head_size=self.head_size,
                    )
                    memory_obj_tensor = memory_obj.get_tensor(i)
                    assert memory_obj_tensor is not None
                    memory_obj_tensor.copy_(tmp_gpu_buffer, non_blocking=True)

        if not memory_obj.raw_tensor.is_cuda:
            # Force a synchronize if the target buffer is NOT CUDA device
            # NOTE: for better performance, we may not want to sync for every
            # memory object
            self.store_stream.synchronize()

        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        with torch.cuda.stream(self.load_stream):
            for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
                self.to_gpu(memory_obj, start, end, **kwargs)
        self.load_stream.synchronize()

    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.from_gpu(memory_obj, start, end, **kwargs)

    def get_shape(self, num_tokens: int) -> torch.Size:
        raise NotImplementedError


class VLLMBufferLayerwiseGPUConnector(GPUConnectorInterface):
    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        use_double_buffer: bool = True,
        **kwargs,
    ):
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers

        self.kvcaches: Optional[List[torch.Tensor]] = None
        self.layout_hints: LayoutHints = (
            kwargs.get(  # type: ignore[assignment]
                "layout_hints"
            )
            or {}
        )

        # TODO(Jiayi): remove this hardcode
        self.cache_positions = True

        self.fused_rotary_emb = None

        assert use_gpu, "use_gpu must be true in VLLMBufferLayerwiseGPUConnector"
        assert "dtype" in kwargs, "dtype should be provided to create a GPU buffer."
        assert "device" in kwargs, "device should be provided to create a GPU buffer."

        self.dtype = kwargs["dtype"]
        self.device = kwargs["device"]

        self.load_stream = torch.cuda.Stream()
        self.store_stream = torch.cuda.Stream()

        self.buffer_mapping: dict[int, MemoryObj] = {}

        # track gap positions between blended chunks
        self.current_gap_positions = None

        self.use_gpu = use_gpu
        self.gpu_buffer_allocator = None
        self.element_size = torch.tensor([], dtype=self.dtype).element_size()

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheMetadata,
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
        layout_hints: Optional[LayoutHints] = None,
    ) -> "VLLMBufferLayerwiseGPUConnector":
        """Create a connector from LMCacheMetadata.

        Args:
            metadata: The LMCache engine metadata containing model configuration.
            use_gpu: Whether to use GPU intermediate buffer.
            device: The device to use for the connector.
            layout_hints: Optional hints about KV cache layout from the
                serving engine.

        Returns:
            A new instance of VLLMBufferLayerwiseGPUConnector.
        """
        # Extract parameters from metadata
        # kv_shape: (num_layer, 2 or 1, chunk_size, num_kv_head, head_size)
        num_layers = metadata.kv_shape[0]
        num_kv_head = metadata.kv_shape[3]
        head_size = metadata.kv_shape[4]
        hidden_dim_size = num_kv_head * head_size

        return cls(
            hidden_dim_size=hidden_dim_size,
            num_layers=num_layers,
            use_gpu=use_gpu,
            dtype=metadata.kv_dtype,
            device=device,
            layout_hints=layout_hints,
        )

    def _lazy_initialize_buffer(self, kv_caches):
        """
        Lazily initialize the GPU buffer allocator if it is not initialized yet.
        Currently, we use the `kv_caches` (kv cache pointer) to determine
        the gpu buffer size in gpu connector.
        Also, the first request might be a bit slower due to buffer creation.
        """
        if self.use_gpu and self.gpu_buffer_allocator is None:
            logger.info("Lazily initializing GPU buffer.")
            # NOTE (Jiayi): We use the first layer to determine the gpu buffer size.
            # NOTE (Jiayi): Using the exact number of tokens in the first layer
            # is okay since fragmentation shouldn't exist in the `gpu_buffer_allocator`
            # in layerwise mode.

            kv_caches = ensure_contiguous_kv_caches(
                kv_caches, kv_layout=self.layout_hints.get("kv_layout")
            )
            self.kvcaches = kv_caches
            self.gpu_kv_format = discover_gpu_kv_format(
                kv_caches, EngineType.VLLM, layout_hints=self.layout_hints
            )
            assert_is_vllm_flash_attn_or_flash_infer(self.gpu_kv_format)
            self.block_size = get_block_size(kv_caches, self.gpu_kv_format)
            self.tokens_per_layer = get_tokens_per_layer(kv_caches, self.gpu_kv_format)
            self.elements_per_layer = get_elements_per_layer(
                kv_caches, self.gpu_kv_format
            )
            logger.info(
                f"Lazily initializing GPU buffer (max tokens={self.tokens_per_layer})."
            )
            # (2026-08-17) 풀 크기가 paged 캡(tokens_per_layer)에 결합돼
            # 있어 KV를 조인 레짐에서 staging이 같이 조여져 고갈-fatal이
            # 난다(SEEDKV_OFF 실측: 55K 프리필 fetch 1건이 57.6K 풀의
            # 95% 점유).  env 하한으로 분리한다(풀은 KV 예산 밖 VRAM).
            _min_tok = int(os.environ.get(
                "LMCACHE_STAGING_MIN_TOKENS", "0") or 0)
            _eff_elems = self.elements_per_layer
            if _min_tok > 0 and self.tokens_per_layer > 0:
                _per_tok = self.elements_per_layer // self.tokens_per_layer
                _eff_elems = max(
                    self.elements_per_layer, _min_tok * _per_tok)
            gpu_buffer_size = _eff_elems * self.element_size
            self.gpu_buffer_allocator = GPUMemoryAllocator(
                gpu_buffer_size, device=self.device
            )

    def get_kv(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the KV cache for the given layer ID.
        This function is used to get the KV cache from the GPU buffer.
        """
        if layer_id not in self.buffer_mapping:
            raise ValueError(f"Layer {layer_id} is not loaded into GPU buffer.")

        gpu_buffer = self.buffer_mapping[layer_id].tensor
        assert gpu_buffer is not None
        return gpu_buffer[0], gpu_buffer[1]

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """ """

        raise NotImplementedError

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """ """

        raise NotImplementedError

    @_lmcache_nvtx_annotate
    def batched_to_gpu(self, starts: List[int], ends: List[int], **kwargs):
        """
        This function is a generator that moves the KV cache from the memory
        objects to buffer GPU memory. In each iteration i, it (1) loads the KV
        cache of layer i from CPU -> GPU buffer, (2) recovers the positional
        encoding of the layer i-1's KV cache in the GPU buffer, and (3)
        moves the KV cache of layer i-2 from GPU buffer to paged GPU memory.
        In total, this the generator will yield num_layers + 2 times.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.
        """

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )
        # Capture the destination KV tensors for this generator. Layerwise
        # loads for multiple requests can coexist in one forward step
        # (SpecAgent scratch fetch plus normal paged LMCache load). The
        # connector object is shared, so a later retriever may overwrite
        # self.kvcaches while this generator is still advancing layer by layer.
        kvcaches = self.kvcaches

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if self.fused_rotary_emb is None and self.cache_positions:
            # TODO(Jiayi): Make this more elegant
            # First Party
            from lmcache.integration.vllm.utils import ENGINE_NAME

            self.lmc_model = LMCBlenderBuilder.get(ENGINE_NAME).layerwise_model
            self.fused_rotary_emb = self.lmc_model.fused_rotary_emb

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        self._lazy_initialize_buffer(kvcaches)

        num_all_tokens = ends[-1] - starts[0]
        slot_mapping_full = slot_mapping[starts[0] : ends[-1]]

        # compute gap positions
        gap_mask = torch.ones(
            num_all_tokens, dtype=torch.bool, device=slot_mapping_full.device
        )
        buf_offset = starts[0]

        for start, end in zip(starts, ends, strict=False):
            gap_mask[start - buf_offset : end - buf_offset] = False

        self.current_gap_positions = torch.where(gap_mask)[0]

        buf_offset = starts[0]
        if self.cache_positions:
            new_positions_full = torch.arange(
                starts[0], ends[-1], dtype=torch.int64, device=kvcaches[0].device
            )

        buffer_shape = self.get_shape(num_all_tokens)
        assert self.gpu_buffer_allocator is not None
        compute_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
            buffer_shape, self.dtype, MemoryFormat.KV_2TD
        )
        load_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
            buffer_shape, self.dtype, MemoryFormat.KV_2TD
        )
        assert compute_gpu_buffer_obj is not None, (
            "Failed to allocate GPU buffer in GPUConnector"
        )
        assert load_gpu_buffer_obj is not None, (
            "Failed to allocate GPU buffer in GPUConnector"
        )
        assert compute_gpu_buffer_obj.tensor is not None
        assert load_gpu_buffer_obj.tensor is not None

        # current_stream = torch.cuda.current_stream()

        if self.cache_positions:
            old_positions_full = torch.zeros(
                (num_all_tokens,), dtype=torch.int64, device=kvcaches[0].device
            )
        for layer_id in range(self.num_layers + 2):
            if layer_id > 1:
                lmc_ops.single_layer_kv_transfer(
                    self.buffer_mapping[layer_id - 2].tensor,
                    kvcaches[layer_id - 2],
                    slot_mapping_full,
                    lmc_ops.TransferDirection.H2D,
                    self.gpu_kv_format,
                    token_major=False,  # shape is [2, num_tokens, hidden_dim]
                )
                del self.buffer_mapping[layer_id - 2]

                logger.debug(f"Finished loading layer {layer_id - 2} into paged memory")

            if layer_id > 0 and layer_id <= self.num_layers:
                # NOTE: wait until both compute and load streams are done
                torch.cuda.synchronize()

                # ping-pong the buffers
                compute_gpu_buffer_obj, load_gpu_buffer_obj = (
                    load_gpu_buffer_obj,
                    compute_gpu_buffer_obj,
                )

                if self.cache_positions:
                    assert compute_gpu_buffer_obj.tensor is not None

                    compute_gpu_buffer_obj.tensor[0] = self.fused_rotary_emb(
                        old_positions_full,
                        new_positions_full,
                        compute_gpu_buffer_obj.tensor[0],
                    )

                # gap zeroing after RoPE
                if self.current_gap_positions.numel():
                    compute_gpu_buffer_obj.tensor[:, self.current_gap_positions] = 0.0

                self.buffer_mapping[layer_id - 1] = compute_gpu_buffer_obj

                logger.debug(f"Finished loading layer {layer_id - 1} into buffer")

            if layer_id < self.num_layers:
                memory_objs_layer = yield

                # memobj -> gpu_buffer
                with torch.cuda.stream(self.load_stream):
                    for start, end, memory_obj in zip(
                        starts, ends, memory_objs_layer, strict=False
                    ):
                        assert memory_obj.metadata.fmt == MemoryFormat.KV_2TD
                        assert load_gpu_buffer_obj.tensor is not None
                        load_gpu_buffer_obj.tensor[0][
                            start - buf_offset : end - buf_offset
                        ].copy_(memory_obj.tensor[0], non_blocking=True)

                        load_gpu_buffer_obj.tensor[1][
                            start - buf_offset : end - buf_offset
                        ].copy_(memory_obj.tensor[1], non_blocking=True)

                        if self.cache_positions and layer_id == 0:
                            old_positions_full[
                                start - buf_offset : end - buf_offset
                            ] = memory_obj.metadata.cached_positions

            elif layer_id == self.num_layers:
                yield

        # free the buffer memory
        load_gpu_buffer_obj.ref_count_down()
        compute_gpu_buffer_obj.ref_count_down()

        assert len(self.buffer_mapping) == 0, (
            "There are still layers in the buffer mapping after "
            "releasing the GPU buffers."
        )

        yield

    # TODO(Jiayi): Reduce repetitive operations in `batched_to_gpu`
    # and `batched_from_gpu`.
    @_lmcache_nvtx_annotate
    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]], List[MemoryObj]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """
        This function is a generator that moves the KV cache from the paged GPU
        memory to the memory objects. The first iteration will prepare some
        related metadata and initiate the transfer in the first layer. In each
        of the following iterations, it will first wait until the storing of
        previous layer finishes, and then initiate string the KV cache of the
        current layer one. The storing process of the KV cache is paged GPU
        memory -> GPU buffer -> memory objects. The last iteration simply waits
        for the last layer to finish.
        In total, this the generator will yield num_layers + 1 times.

        :param memory_objs: The memory objects to store the KV cache. The first
            dimension is the number of layers, and the second dimension is the
            number of memory objects (i.e., number of chunks) for each layer.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'kvcaches' is not provided in kwargs.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )
        # Capture the destination KV tensors for this generator. Layerwise
        # loads for multiple requests can coexist in one forward step
        # (SpecAgent scratch fetch plus normal paged LMCache load). The
        # connector object is shared, so a later retriever may overwrite
        # self.kvcaches while this generator is still advancing layer by layer.
        kvcaches = self.kvcaches

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        self._lazy_initialize_buffer(kvcaches)

        buf_start = 0
        slot_mapping_chunks = []
        buf_starts_ends = []
        old_positions_chunks = []
        for start, end in zip(starts, ends, strict=False):
            buf_end = buf_start + end - start
            buf_starts_ends.append((buf_start, buf_end))
            slot_mapping_chunks.append(slot_mapping[start:end])
            buf_start = buf_end
            if self.cache_positions:
                old_positions_chunks.append(
                    torch.arange(
                        start, end, device=kvcaches[0].device, dtype=torch.int64
                    )
                )

        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)
        buffer_shape = self.get_shape(num_tokens)
        assert self.gpu_buffer_allocator is not None
        tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
            buffer_shape, self.dtype, MemoryFormat.KV_2TD
        )
        assert tmp_gpu_buffer_obj is not None, (
            "Failed to allocate GPU buffer in GPUConnector"
        )
        assert tmp_gpu_buffer_obj.tensor is not None

        current_stream = torch.cuda.current_stream()
        if kwargs.get("specagent_save_current_layer_only", False):
            for layer_id in range(self.num_layers):
                memory_objs_layer = memory_objs[layer_id]
                with torch.cuda.stream(self.store_stream):
                    self.store_stream.wait_stream(current_stream)
                    lmc_ops.single_layer_kv_transfer(
                        tmp_gpu_buffer_obj.tensor,
                        kvcaches[layer_id],
                        slot_mapping_full,
                        lmc_ops.TransferDirection.D2H,
                        self.gpu_kv_format,
                        token_major=False,
                    )
                    for (buf_start, buf_end), memory_obj, old_positions in zip(
                        buf_starts_ends,
                        memory_objs_layer,
                        old_positions_chunks,
                        strict=False,
                    ):
                        assert memory_obj.tensor is not None
                        memory_obj.tensor[0].copy_(
                            tmp_gpu_buffer_obj.tensor[0][buf_start:buf_end],
                            non_blocking=True,
                        )
                        memory_obj.tensor[1].copy_(
                            tmp_gpu_buffer_obj.tensor[1][buf_start:buf_end],
                            non_blocking=True,
                        )
                        if self.cache_positions:
                            memory_obj.metadata.cached_positions = old_positions

                self.store_stream.synchronize()
                logger.debug(f"Finished offloading layer {layer_id}")
                yield

            tmp_gpu_buffer_obj.ref_count_down()
            yield
            return

        for layer_id in range(self.num_layers):
            memory_objs_layer = memory_objs[layer_id]
            # kvcaches -> gpu_buffer -> memobj
            with torch.cuda.stream(self.store_stream):
                self.store_stream.wait_stream(current_stream)
                lmc_ops.single_layer_kv_transfer(
                    tmp_gpu_buffer_obj.tensor,
                    kvcaches[layer_id],
                    slot_mapping_full,
                    lmc_ops.TransferDirection.D2H,
                    self.gpu_kv_format,
                    token_major=False,  # shape is [2, num_tokens, hidden_dim]
                )
                for (buf_start, buf_end), memory_obj, old_positions in zip(
                    buf_starts_ends,
                    memory_objs_layer,
                    old_positions_chunks,
                    strict=False,
                ):
                    assert memory_obj.tensor is not None
                    memory_obj.tensor[0].copy_(
                        tmp_gpu_buffer_obj.tensor[0][buf_start:buf_end],
                        non_blocking=True,
                    )
                    memory_obj.tensor[1].copy_(
                        tmp_gpu_buffer_obj.tensor[1][buf_start:buf_end],
                        non_blocking=True,
                    )
                    if self.cache_positions:
                        memory_obj.metadata.cached_positions = old_positions

            yield
            self.store_stream.synchronize()
            logger.debug(f"Finished offloading layer {layer_id}")

        # free the buffer memory
        tmp_gpu_buffer_obj.ref_count_down()
        yield

    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size([2, num_tokens, self.hidden_dim_size])


class VLLMPagedMemLayerwiseGPUConnector(GPUConnectorInterface):
    """ """

    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        **kwargs,
    ):
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.use_gpu = use_gpu
        self.layout_hints: LayoutHints = (
            kwargs.get(  # type: ignore[assignment]
                "layout_hints"
            )
            or {}
        )

        self.gpu_buffer_allocator = None

        assert "chunk_size" in kwargs, (
            "chunk_size should be provided to create a GPU buffer."
        )
        assert "dtype" in kwargs, "dtype should be provided to create a GPU buffer."
        assert "device" in kwargs, "device should be provided to create a GPU buffer."

        self.dtype = kwargs["dtype"]
        self.device = kwargs["device"]

        self.kvcaches: Optional[List[torch.Tensor]] = None

        # All sizes are in bytes
        self.element_size = torch.tensor([], dtype=self.dtype).element_size()

        if os.environ.get("SPECAGENT_LOW_PRIORITY_LOAD_STREAM"):
            least_priority, _ = torch.cuda.Stream.priority_range()
            self.load_stream = torch.cuda.Stream(priority=least_priority)
            logger.info(
                "[SpecAgent][LMC-STREAM] using low-priority layerwise "
                "load stream priority=%s",
                least_priority,
            )
        else:
            self.load_stream = torch.cuda.Stream()
        self.store_stream = torch.cuda.Stream()
        self._specagent_pending_store_buffer_frees: list[
            tuple[torch.cuda.Event, MemoryObj]
        ] = []
        # Plans from all SpecAgent scratch retrievers are collected per layer
        # and submitted together by the adapter.  CPU memory and the existing
        # per-retriever temporary GPU buffers remain owned by their generators;
        # this queue adds descriptors only.
        self._specagent_deferred_layer_fetches: dict[int, list[dict]] = {}
        # A deferred submission is genuinely asynchronous.  Keep every tensor
        # descriptor (in particular pinned CPU source views and slot mappings)
        # alive until the load-stream event says that both H2D and scatter are
        # finished.  The retriever generator may otherwise advance/finish and
        # drop the last Python reference while CUDA still consumes the memory.
        self._specagent_pending_deferred_plans: list[
            tuple[torch.cuda.Event, list[dict]]
        ] = []
        self._specagent_fused_debug_checked_layers: set[int] = set()

        self.use_mla = "use_mla" in kwargs and kwargs["use_mla"]

    def _specagent_log_staging_physical(
        self,
        action: str,
        kind: str,
        num_tokens: int,
        success: bool = True,
    ) -> None:
        """Log actual staging-pool ownership without changing allocation.

        Scheduler reservations finish with their model futures, while a
        layerwise CUDA copy may retain its temporary allocation longer.  This
        opt-in trace exposes that lifetime gap so defer decisions can be
        audited against physical capacity rather than inferred from the
        scheduler ledger.
        """
        if not os.environ.get("LMCACHE_ALLOC_TRACE"):
            return
        gpu_allocator = self.gpu_buffer_allocator
        allocator = getattr(gpu_allocator, "allocator", None)
        address_manager = getattr(allocator, "address_manager", None)
        allocated_bytes = int(getattr(
            address_manager, "total_allocated_size", 0,
        ) or 0)
        pool_bytes = int(getattr(
            address_manager, "_size", 0,
        ) or 0)
        if pool_bytes <= 0:
            buffer = getattr(allocator, "buffer", None)
            pool_bytes = int(buffer.numel()) if buffer is not None else 0
        per_token_bytes = int(self.hidden_dim_size) * int(self.element_size)
        if not self.use_mla:
            per_token_bytes *= 2
        allocated_tokens = (
            allocated_bytes // per_token_bytes if per_token_bytes else 0
        )
        pool_tokens = pool_bytes // per_token_bytes if per_token_bytes else 0
        logger.info(
            "[SpecAgent][STAGING-PHYSICAL] action=%s kind=%s tokens=%d "
            "success=%s allocated_tokens=%d pool_tokens=%d active=%d "
            "pending_load_plans=%d pending_store_frees=%d",
            action,
            kind,
            int(num_tokens),
            bool(success),
            allocated_tokens,
            pool_tokens,
            int(getattr(allocator, "num_active_allocations", 0) or 0),
            len(self._specagent_pending_deferred_plans),
            len(self._specagent_pending_store_buffer_frees),
        )

    def specagent_has_deferred_layer(
        self,
        layer_id: int,
        layer_events: object,
    ) -> bool:
        """Whether this retriever has queued its plan for ``layer_id``.

        Deferred plans intentionally do not publish their CUDA ready event
        until all retrievers for the layer are flushed together.  The adapter
        therefore cannot use ``events[layer_id] is None`` as its generator
        advancement condition: doing so drains later layers into the queue
        and reuses the retriever's temporary double-buffer slot too early.
        """
        return any(
            plan.get("layer_events") is layer_events
            for plan in self._specagent_deferred_layer_fetches.get(
                int(layer_id), ()
            )
        )

    def specagent_flush_deferred_layer(
        self,
        layer_id: int,
        scratch_free_event: object | None = None,
    ) -> bool:
        still_pending: list[tuple[torch.cuda.Event, list[dict]]] = []
        for event, pending_plans in self._specagent_pending_deferred_plans:
            if event.query():
                # The generator deliberately transferred ownership of its
                # temporary allocation to the last-layer plan.  Return it to
                # the allocator only after the last GPU consumer has finished.
                for pending_plan in pending_plans:
                    release_obj = pending_plan.get("release_obj")
                    if release_obj is not None:
                        release_tokens = int(release_obj.tensor.shape[0])
                        release_obj.ref_count_down()
                        self._specagent_log_staging_physical(
                            "free", "load", release_tokens,
                        )
            else:
                still_pending.append((event, pending_plans))
        self._specagent_pending_deferred_plans = still_pending
        plans = self._specagent_deferred_layer_fetches.pop(
            int(layer_id), None
        )
        if plans and os.environ.get("SPECAGENT_FUSED_TRANSFER_DEBUG"):
            logger.error(
                "[SpecAgent][FUSED-FLUSH] layer=%d plans=%d event=%s "
                "connector_id=%d",
                layer_id,
                len(plans) if plans else 0,
                scratch_free_event is not None,
                id(self),
            )
        if not plans:
            return False

        if (
            int(layer_id) == 0
            and os.environ.get("SPECAGENT_FUSED_MAPPING_DEBUG")
        ):
            for plan_idx, plan in enumerate(plans):
                mapping = plan["slot_mapping"]
                transfer = plan["transfer_tensor"]
                target = plan["target_kv_cache"]
                mapping_min = int(mapping.min().item()) if mapping.numel() else 0
                mapping_max = int(mapping.max().item()) if mapping.numel() else -1
                if transfer.dim() != 3:
                    raise RuntimeError(
                        "[SpecAgent][FUSED-MAP-ERROR] unexpected transfer "
                        f"shape={tuple(transfer.shape)}"
                    )
                transfer_tokens = int(
                    transfer.shape[1]
                    if transfer.shape[0] == 2
                    else transfer.shape[0]
                )
                if self.use_mla:
                    target_slots = int(target.shape[0] * target.shape[1])
                elif is_hnd(self.gpu_kv_format):
                    target_slots = int(target.shape[1] * target.shape[3])
                else:
                    target_slots = int(target.shape[1] * target.shape[2])
                logger.error(
                    "[SpecAgent][FUSED-MAP] plan=%d mapping=%d range=[%d,%d] "
                    "transfer_tokens=%d target_slots=%d target_shape=%s",
                    plan_idx,
                    int(mapping.numel()),
                    mapping_min,
                    mapping_max,
                    transfer_tokens,
                    target_slots,
                    tuple(target.shape),
                )
                if mapping_min < -1 or mapping_max >= target_slots:
                    raise RuntimeError(
                        "[SpecAgent][FUSED-MAP-ERROR] slot mapping outside "
                        f"scratch target: range=[{mapping_min},{mapping_max}] "
                        f"target_slots={target_slots}"
                    )
                if int(mapping.numel()) > transfer_tokens:
                    raise RuntimeError(
                        "[SpecAgent][FUSED-MAP-ERROR] mapping longer than "
                        f"transfer buffer: mapping={mapping.numel()} "
                        f"transfer_tokens={transfer_tokens}"
                    )

        with torch.cuda.stream(self.load_stream):
            copy_dsts = [
                dst for plan in plans for dst in plan["copy_dsts"]
            ]
            copy_srcs = [
                src for plan in plans for src in plan["copy_srcs"]
            ]
            if copy_dsts:
                if all(
                    _can_raw_h2d_copy(dst, src)
                    for dst, src in zip(
                        copy_dsts, copy_srcs, strict=False
                    )
                ):
                    _raw_h2d_copy_batch_async(copy_dsts, copy_srcs)
                else:
                    # Preserve the existing correctness fallback for unusual
                    # storage backends or tensor layouts.
                    for plan in plans:
                        torch._foreach_copy_(
                            plan["copy_dsts"],
                            plan["copy_srcs"],
                            non_blocking=True,
                        )

            # H2D targets only the existing per-retriever temporary buffers,
            # so it can run while the current layer reads scratch.  Delay only
            # the writes into the two-slot scratch ring.
            if scratch_free_event is not None:
                self.load_stream.wait_event(scratch_free_event)
            lmc_ops.specagent_multi_scatter(
                [plan["transfer_tensor"] for plan in plans],
                [plan["target_kv_cache"] for plan in plans],
                [plan["slot_mapping"] for plan in plans],
                self.gpu_kv_format,
            )
            ready = _specagent_record_cuda_event(self.load_stream)

        for plan in plans:
            events = plan["layer_events"]
            if int(layer_id) < len(events):
                events[int(layer_id)] = ready
        self._specagent_pending_deferred_plans.append((ready, plans))

        # Debug-only, exact payload validation.  Full-model recomputation is
        # not bit-identical for this path, so validate the two operations that
        # the fused submit changes instead: pinned CPU -> temporary GPU tensor,
        # then temporary tensor -> paged scratch KV.  Synchronization and
        # device-to-host copies are intentionally confined to this opt-in path.
        if os.environ.get("SPECAGENT_FUSED_TRANSFER_DEBUG"):
            layer_filter_raw = os.environ.get(
                "SPECAGENT_FUSED_TRANSFER_DEBUG_LAYERS", "0,1,45"
            )
            if layer_filter_raw.strip().lower() == "all":
                layer_filter = set(range(self.num_layers))
            else:
                layer_filter = {
                    int(value)
                    for value in layer_filter_raw.split(",")
                    if value.strip()
                }
            if (
                int(layer_id) in layer_filter
                and int(layer_id)
                not in self._specagent_fused_debug_checked_layers
            ):
                ready.synchronize()
                for plan_idx, plan in enumerate(plans):
                    h2d_mismatch = 0
                    h2d_max = 0.0
                    for dst, src in zip(
                        plan["copy_dsts"],
                        plan["copy_srcs"],
                        strict=True,
                    ):
                        actual = dst.detach().cpu()
                        expected = src.detach().cpu()
                        h2d_mismatch += int(
                            torch.count_nonzero(actual != expected).item()
                        )
                        if actual.numel():
                            h2d_max = max(
                                h2d_max,
                                float(
                                    (
                                        actual.float()
                                        - expected.float()
                                    ).abs().max().item()
                                ),
                            )

                    transfer = plan["transfer_tensor"]
                    target = plan["target_kv_cache"]
                    mapping = plan["slot_mapping"].to(
                        device=transfer.device, dtype=torch.long
                    )
                    block_size = int(target.shape[2])
                    scatter_mismatch = 0
                    scatter_max = 0.0
                    checked_tokens = 0
                    for begin in range(0, int(mapping.numel()), 256):
                        slots = mapping[begin : begin + 256]
                        valid = slots >= 0
                        if not bool(valid.any().item()):
                            continue
                        token_indices = (
                            torch.arange(
                                begin,
                                begin + int(slots.numel()),
                                device=slots.device,
                            )[valid]
                        )
                        slots = slots[valid]
                        blocks = torch.div(
                            slots, block_size, rounding_mode="floor"
                        )
                        offsets = slots.remainder(block_size)
                        if transfer.dim() == 3 and transfer.shape[0] == 2:
                            expected = transfer[:, token_indices, :]
                        elif transfer.dim() == 3 and transfer.shape[1] == 2:
                            expected = transfer[token_indices, :, :].permute(
                                1, 0, 2
                            )
                        else:
                            raise AssertionError(
                                "Unexpected transfer tensor shape "
                                f"{tuple(transfer.shape)}"
                            )
                        actual = target[:, blocks, offsets].reshape(
                            2, int(slots.numel()), -1
                        )
                        expected = expected.reshape(
                            2, int(slots.numel()), -1
                        )
                        scatter_mismatch += int(
                            torch.count_nonzero(actual != expected).item()
                        )
                        if actual.numel():
                            scatter_max = max(
                                scatter_max,
                                float(
                                    (
                                        actual.float()
                                        - expected.float()
                                    ).abs().max().item()
                                ),
                            )
                        checked_tokens += int(slots.numel())
                    logger.error(
                        "[SpecAgent][FUSED-CHECK] layer=%d plan=%d "
                        "copies=%d tokens=%d h2d_mismatch=%d h2d_max=%.6g "
                        "scatter_mismatch=%d scatter_max=%.6g",
                        layer_id,
                        plan_idx,
                        len(plan["copy_dsts"]),
                        checked_tokens,
                        h2d_mismatch,
                        h2d_max,
                        scatter_mismatch,
                        scatter_max,
                    )
                self._specagent_fused_debug_checked_layers.add(int(layer_id))
        return True

    def _specagent_release_completed_store_buffers(self) -> None:
        if not self._specagent_pending_store_buffer_frees:
            return
        pending: list[tuple[torch.cuda.Event, MemoryObj]] = []
        for event, memory_obj in self._specagent_pending_store_buffer_frees:
            if event.query():
                release_tokens = int(memory_obj.tensor.shape[0])
                memory_obj.ref_count_down()
                self._specagent_log_staging_physical(
                    "free", "store", release_tokens,
                )
            else:
                pending.append((event, memory_obj))
        self._specagent_pending_store_buffer_frees = pending

    def _specagent_defer_store_buffer_free(self, memory_obj: MemoryObj) -> None:
        with torch.cuda.stream(self.store_stream):
            event = _specagent_record_cuda_event(self.store_stream)
        self._specagent_pending_store_buffer_frees.append((event, memory_obj))

    def _specagent_wait_one_store_buffer(self) -> None:
        if not self._specagent_pending_store_buffer_frees:
            return
        event, memory_obj = self._specagent_pending_store_buffer_frees.pop(0)
        event.synchronize()
        release_tokens = int(memory_obj.tensor.shape[0])
        memory_obj.ref_count_down()
        self._specagent_log_staging_physical(
            "free_wait", "store", release_tokens,
        )
        self._specagent_release_completed_store_buffers()

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheMetadata,
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
        layout_hints: Optional[LayoutHints] = None,
    ) -> "VLLMPagedMemLayerwiseGPUConnector":
        """Create a connector from LMCacheMetadata.

        Args:
            metadata: The LMCache engine metadata containing model configuration.
            use_gpu: Whether to use GPU intermediate buffer.
            device: The device to use for the connector.
            layout_hints: Optional hints about KV cache layout from the
                serving engine.

        Returns:
            A new instance of VLLMPagedMemLayerwiseGPUConnector.
        """
        # Extract parameters from metadata
        # kv_shape: (num_layer, 2 or 1, chunk_size, num_kv_head, head_size)
        num_layers = metadata.kv_shape[0]
        chunk_size = metadata.kv_shape[2]
        num_kv_head = metadata.kv_shape[3]
        head_size = metadata.kv_shape[4]
        hidden_dim_size = num_kv_head * head_size

        return cls(
            hidden_dim_size=hidden_dim_size,
            num_layers=num_layers,
            use_gpu=use_gpu,
            chunk_size=chunk_size,
            dtype=metadata.kv_dtype,
            device=device,
            use_mla=metadata.use_mla,
            layout_hints=layout_hints,
        )

    def _lazy_initialize_buffer(self, kv_caches):
        """
        Lazily initialize the GPU buffer allocator if it is not initialized yet.
        Currently, we use the `kv_caches` (kv cache pointer) to determine
        the gpu buffer size in gpu connector.
        Also, the first request might be a bit slower due to buffer creation.
        """
        if self.use_gpu and self.gpu_buffer_allocator is None:
            logger.info("Lazily initializing GPU buffer.")
            # NOTE (Jiayi): We use the first layer to determine the gpu buffer size.
            # NOTE (Jiayi): Using the exact number of tokens in the first layer
            # is okay since fragmentation shouldn't exist in the `gpu_buffer_allocator`
            # in layerwise mode.

            kv_caches = ensure_contiguous_kv_caches(
                kv_caches, kv_layout=self.layout_hints.get("kv_layout")
            )
            self.kvcaches = kv_caches
            self.gpu_kv_format = discover_gpu_kv_format(
                kv_caches, EngineType.VLLM, layout_hints=self.layout_hints
            )
            assert_is_vllm_flash_attn_or_flash_infer(self.gpu_kv_format)
            self.block_size = get_block_size(kv_caches, self.gpu_kv_format)
            self.tokens_per_layer = get_tokens_per_layer(kv_caches, self.gpu_kv_format)
            self.elements_per_layer = get_elements_per_layer(
                kv_caches, self.gpu_kv_format
            )
            logger.info(
                f"Lazily initializing GPU buffer (max tokens={self.tokens_per_layer})."
            )
            # (2026-08-17) 풀 크기가 paged 캡(tokens_per_layer)에 결합돼
            # 있어 KV를 조인 레짐에서 staging이 같이 조여져 고갈-fatal이
            # 난다(SEEDKV_OFF 실측: 55K 프리필 fetch 1건이 57.6K 풀의
            # 95% 점유).  env 하한으로 분리한다(풀은 KV 예산 밖 VRAM).
            _min_tok = int(os.environ.get(
                "LMCACHE_STAGING_MIN_TOKENS", "0") or 0)
            _eff_elems = self.elements_per_layer
            if _min_tok > 0 and self.tokens_per_layer > 0:
                _per_tok = self.elements_per_layer // self.tokens_per_layer
                _eff_elems = max(
                    self.elements_per_layer, _min_tok * _per_tok)
            gpu_buffer_size = _eff_elems * self.element_size
            self.gpu_buffer_allocator = GPUMemoryAllocator(
                gpu_buffer_size, device=self.device
            )

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """ """

        raise NotImplementedError

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """ """

        raise NotImplementedError

    @_lmcache_nvtx_annotate
    def batched_to_gpu(self, starts: List[int], ends: List[int], **kwargs):
        """
        This function is a generator that moves the KV cache from the memory
        objects to paged GPU memory. The first iteration will prepare some
        related metadata. In each of the following iterations, it will first
        wait until the loading of the previous layer finish, and then load
        one layer of KV cache from the memory objects -> GPU buffer ->
        paged GPU memory. The last iteration simply waits for the last layer
        to finish.
        In total, this the generator will yield num_layers + 2 times.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """

        self._specagent_release_completed_store_buffers()
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )
        # Keep this generator bound to the destination passed at creation.
        # Another concurrent layerwise retriever may later repoint
        # self.kvcaches to a different destination.
        kvcaches = self.kvcaches

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        sync: bool = kwargs["sync"]
        offset: int = int(kwargs.get("offset", 0) or 0)
        write_token_mask: Optional[torch.Tensor] = kwargs.get(
            "specagent_write_token_mask"
        )
        write_token_mask_cpu: Optional[torch.Tensor] = None
        if write_token_mask is not None:
            if len(write_token_mask) != len(slot_mapping):
                raise ValueError(
                    "specagent_write_token_mask must match slot_mapping "
                    f"length, got mask={len(write_token_mask)} "
                    f"slot_mapping={len(slot_mapping)}"
                )
            write_token_mask_cpu = write_token_mask.to(
                device="cpu", dtype=torch.bool
            )
        defer_tmp_buffer_free = bool(
            kwargs.get("specagent_defer_store_tmp_buffer_free", True)
        )

        self._lazy_initialize_buffer(kvcaches)

        slot_mapping_chunks = []
        write_mask_chunks = []
        for start, end in zip(starts, ends, strict=False):
            local_start = start - offset
            local_end = end - offset
            if local_start < 0 or local_end > len(slot_mapping):
                raise ValueError(
                    "Invalid layerwise store range after offset adjustment: "
                    f"start={start} end={end} offset={offset} "
                    f"slot_mapping_len={len(slot_mapping)}"
                )
            slot_mapping_chunks.append(slot_mapping[local_start:local_end])
            if write_token_mask_cpu is not None:
                write_mask_chunks.append(write_token_mask_cpu[
                    local_start:local_end
                ])

        copy_plan: Optional[list[tuple[int, int, int, int, int]]] = None
        if write_mask_chunks:
            copy_plan = []
            compact_slot_mapping_chunks = []
            dst_offset = 0
            for chunk_idx, (slot_mapping_chunk, mask_chunk) in enumerate(
                zip(slot_mapping_chunks, write_mask_chunks, strict=False)
            ):
                true_positions = torch.nonzero(mask_chunk, as_tuple=False).flatten()
                if true_positions.numel() == 0:
                    continue
                positions = [int(pos) for pos in true_positions.tolist()]
                run_start = positions[0]
                prev = positions[0]
                for pos in positions[1:]:
                    if pos != prev + 1:
                        run_end = prev + 1
                        run_len = run_end - run_start
                        copy_plan.append((
                            chunk_idx,
                            run_start,
                            run_end,
                            dst_offset,
                            dst_offset + run_len,
                        ))
                        compact_slot_mapping_chunks.append(
                            slot_mapping_chunk[run_start:run_end]
                        )
                        dst_offset += run_len
                        run_start = pos
                    prev = pos
                run_end = prev + 1
                run_len = run_end - run_start
                copy_plan.append((
                    chunk_idx,
                    run_start,
                    run_end,
                    dst_offset,
                    dst_offset + run_len,
                ))
                compact_slot_mapping_chunks.append(
                    slot_mapping_chunk[run_start:run_end]
                )
                dst_offset += run_len
            slot_mapping_full = (
                torch.cat(compact_slot_mapping_chunks, dim=0)
                if compact_slot_mapping_chunks
                else slot_mapping.new_empty((0,))
            )
        else:
            # TODO(Jiayi): Optimize away this `cat`
            slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)

        tmp_gpu_buffer_obj: Optional[MemoryObj] = None
        if self.use_gpu:
            buffer_shape = self.get_shape(num_tokens)
            assert self.gpu_buffer_allocator is not None
            self._specagent_release_completed_store_buffers()
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, MemoryFormat.KV_T2D
            )
            self._specagent_log_staging_physical(
                "alloc", "load", num_tokens,
                success=tmp_gpu_buffer_obj is not None,
            )
            if tmp_gpu_buffer_obj is None:
                self._specagent_wait_one_store_buffer()
                tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                    buffer_shape, self.dtype, MemoryFormat.KV_T2D
                )
                self._specagent_log_staging_physical(
                    "alloc_retry", "load", num_tokens,
                    success=tmp_gpu_buffer_obj is not None,
                )
            if tmp_gpu_buffer_obj is None:
                # Transient exhaustion: concurrent verify fetches can briefly
                # hold the whole staging pool (observed 2x on 2026-07-26 at
                # 24-30 concurrent requests; the previous assert killed the
                # engine).  Buffers free asynchronously as load-stream events
                # complete, so a bounded retry resolves it without new memory.
                _alloc_deadline = time.monotonic() + 5.0
                while tmp_gpu_buffer_obj is None:
                    self._specagent_release_completed_store_buffers()
                    self.specagent_flush_deferred_layer(-1)  # reap done plans
                    tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                        buffer_shape, self.dtype, MemoryFormat.KV_T2D
                    )
                    if tmp_gpu_buffer_obj is not None:
                        break
                    if time.monotonic() > _alloc_deadline:
                        raise RuntimeError(
                            "GPU staging buffer exhausted for >5s "
                            f"(need shape={buffer_shape}); concurrent "
                            "verify fetches exceed the staging pool"
                        )
                    time.sleep(0.002)
            assert tmp_gpu_buffer_obj is not None, (
                "Failed to allocate GPU buffer in GPUConnector"
            )
            assert tmp_gpu_buffer_obj.tensor is not None

        # foreach-alt: precompute layer-invariant dst views into the tmp buffer
        # once (chunk lengths from slot_mapping_chunks). Removes per-chunk Python
        # (zip/bounds/assert/slice/offset) from the per-layer hot loop.
        _fa_dsts: Optional[list] = None
        if self.use_gpu:
            _fa_dsts = []
            if copy_plan is not None:
                for _, _, _, dst_start, dst_end in copy_plan:
                    _fa_dsts.append(tmp_gpu_buffer_obj.tensor[dst_start:dst_end])
            else:
                _fa_off = 0
                for _smc in slot_mapping_chunks:
                    _clen = len(_smc)
                    _fa_dsts.append(
                        tmp_gpu_buffer_obj.tensor[_fa_off : _fa_off + _clen]
                    )
                    _fa_off += _clen

        current_stream = torch.cuda.current_stream()
        debug_transfer = bool(os.environ.get("SPECAGENT_LMCACHE_TRANSFER_DEBUG"))
        debug_limit = int(os.environ.get("SPECAGENT_LMCACHE_TRANSFER_DEBUG_MAX", "8"))
        debug_target_token_raw = os.environ.get(
            "SPECAGENT_LMCACHE_TRANSFER_DEBUG_TOKEN")
        debug_target_token = (
            int(debug_target_token_raw)
            if debug_target_token_raw not in (None, "") else None
        )
        debug_count = 0
        bw_profile = bool(os.environ.get("SPECAGENT_BW_PROFILE"))
        bw_profile_records = []
        specagent_layer_events = kwargs.get("specagent_layer_events")
        event_synchronized_load = isinstance(specagent_layer_events, list)
        defer_layer_submit = bool(
            kwargs.get("specagent_defer_layer_submit")
            and self.use_gpu
            and event_synchronized_load
        )
        if os.environ.get("SPECAGENT_FUSED_TRANSFER_DEBUG"):
            logger.error(
                "[SpecAgent][FUSED-CONNECTOR] requested=%s use_gpu=%s "
                "event_synchronized=%s defer=%s starts=%d",
                bool(kwargs.get("specagent_defer_layer_submit")),
                self.use_gpu,
                event_synchronized_load,
                defer_layer_submit,
                len(starts),
            )
        if defer_layer_submit:
            # Timing events and payload checks belong around the eventual
            # fused flush, not descriptor collection.
            bw_profile = False

        _hb = bool(os.environ.get("SPECAGENT_HOST_BREAKDOWN"))
        import time as _hbt
        _hb_copy = _hb_xfer = 0.0
        for layer_id in range(self.num_layers):
            memory_objs_layer = yield
            _hb_t0 = _hbt.perf_counter() if _hb else 0.0
            if sync and not event_synchronized_load:
                current_stream.wait_stream(self.load_stream)
            if layer_id > 0:
                logger.debug(f"Finished loading layer {layer_id - 1}")

            # memobj -> gpu_buffer -> kvcaches
            prof_start = prof_end = None
            prof_h2d_bytes = 0
            prof_gpu_copy_bytes = 0
            if bw_profile:
                prof_end = True
            with torch.cuda.stream(self.load_stream):
                if bw_profile:
                    prof_start = _specagent_record_cuda_event(
                        self.load_stream, enable_timing=True
                    )
                buffer_offset = 0
                debug_first: Optional[tuple[int, int, int, int]] = None
                debug_targets: list[tuple[int, int, int, int, int]] = []
                if self.use_gpu:
                    # fast path: dst views precomputed (_fa_dsts); fill tmp buffer
                    # with one batched multi-tensor copy. No per-chunk Python in
                    # the hot loop (validation already done pre-loop; fmt checked
                    # once on layer 0).
                    if layer_id == 0 and memory_objs_layer:
                        _exp_fmt = (
                            MemoryFormat.KV_MLA_FMT if self.use_mla
                            else MemoryFormat.KV_T2D
                        )
                        assert memory_objs_layer[0].metadata.fmt == _exp_fmt, (
                            f"Expected memory format {_exp_fmt}, got "
                            f"{memory_objs_layer[0].metadata.fmt}"
                        )
                    with _lmc_record(
                        "lmcache gpu_send: build_srcs "
                        f"layer={layer_id} chunks={len(memory_objs_layer)} "
                        f"copy_plan={copy_plan is not None}"
                    ):
                        if copy_plan is not None:
                            _fa_srcs = []
                            for (
                                chunk_idx,
                                src_start,
                                src_end,
                                _dst_start,
                                _dst_end,
                            ) in copy_plan:
                                src = memory_objs_layer[chunk_idx].tensor
                                if src_start == 0 and src_end == src.shape[0]:
                                    _fa_srcs.append(src)
                                else:
                                    _fa_srcs.append(src[src_start:src_end])
                        else:
                            _fa_srcs = [mo.tensor for mo in memory_objs_layer]
                    _fa_n = min(len(_fa_dsts), len(_fa_srcs))
                    if _fa_n and not defer_layer_submit:
                        with _lmc_record(
                            "lmcache gpu_send: h2d_copy "
                            f"layer={layer_id} copies={_fa_n}"
                        ):
                            _copy_dsts = _fa_dsts[:_fa_n]
                            _copy_srcs = _fa_srcs[:_fa_n]
                            if all(
                                _can_raw_h2d_copy(_dst, _src)
                                for _dst, _src in zip(
                                    _copy_dsts, _copy_srcs, strict=False
                                )
                            ):
                                _raw_h2d_copy_batch_async(
                                    _copy_dsts,
                                    _copy_srcs,
                                )
                            else:
                                torch._foreach_copy_(
                                    _copy_dsts,
                                    _copy_srcs,
                                    non_blocking=True,
                                )
                    if bw_profile:
                        for _src in _fa_srcs:
                            prof_h2d_bytes += int(
                                _src.numel() * _src.element_size()
                            )
                else:
                    for start, end, memory_obj in zip(
                        starts, ends, memory_objs_layer, strict=False
                    ):
                        local_start = start - offset
                        local_end = end - offset
                        lmc_ops.single_layer_kv_transfer(
                            memory_obj.tensor,
                            kvcaches[layer_id],
                            slot_mapping[local_start:local_end],
                            lmc_ops.TransferDirection.H2D,
                            self.gpu_kv_format,
                            token_major=True,
                        )

                if _hb:
                    _hb_c1 = _hbt.perf_counter(); _hb_copy += _hb_c1 - _hb_t0
                if self.use_gpu:
                    transfer_tensor = tmp_gpu_buffer_obj.tensor
                    transfer_slot_mapping = slot_mapping_full
                    if defer_layer_submit:
                        self._specagent_deferred_layer_fetches.setdefault(
                            int(layer_id), []
                        ).append({
                            "copy_dsts": list(_fa_dsts[:_fa_n]),
                            "copy_srcs": list(_fa_srcs[:_fa_n]),
                            "transfer_tensor": transfer_tensor,
                            "target_kv_cache": kvcaches[layer_id],
                            "slot_mapping": transfer_slot_mapping,
                            "layer_events": specagent_layer_events,
                            "release_obj": (
                                tmp_gpu_buffer_obj
                                if layer_id == self.num_layers - 1
                                else None
                            ),
                        })
                    else:
                        with _lmc_record(
                            "lmcache gpu_send: xfer_to_kvcache "
                            f"layer={layer_id} tokens={num_tokens}"
                        ):
                            lmc_ops.single_layer_kv_transfer(
                                transfer_tensor,
                                kvcaches[layer_id],
                                transfer_slot_mapping,
                                lmc_ops.TransferDirection.H2D,
                                self.gpu_kv_format,
                                token_major=True,
                            )
                    if _hb:
                        _hb_xfer += _hbt.perf_counter() - _hb_c1
                    if bw_profile:
                        prof_gpu_copy_bytes = int(
                            tmp_gpu_buffer_obj.tensor.numel()
                            * tmp_gpu_buffer_obj.tensor.element_size()
                        )
                    if (not defer_layer_submit
                            and debug_transfer and debug_count < debug_limit
                            and debug_first is not None):
                        debug_points = [
                            ("CHECK", -1, *debug_first),
                            *[
                                ("TARGET", token, buf_idx, slot, block, block_offset)
                                for token, buf_idx, slot, block, block_offset
                                in debug_targets
                            ],
                        ]
                        for (
                            debug_kind,
                            token,
                            buf_idx,
                            slot,
                            block,
                            block_offset,
                        ) in debug_points:
                            try:
                                buf = tmp_gpu_buffer_obj.tensor
                                if buf.dim() == 3 and buf.shape[0] == 2:
                                    src = buf[:, buf_idx, :].reshape(-1).float()
                                elif buf.dim() == 3 and buf.shape[1] == 2:
                                    src = buf[buf_idx, :, :].reshape(-1).float()
                                else:
                                    src = buf.reshape(-1).float()[
                                        buf_idx * 2 * self.hidden_dim_size:
                                        (buf_idx + 1) * 2 * self.hidden_dim_size
                                    ]
                                dst = kvcaches[layer_id][
                                    :, block, block_offset
                                ].reshape(-1).float()
                                max_delta = float((src - dst).abs().max().item())
                                src_sum = float(src.sum().item())
                                dst_sum = float(dst.sum().item())
                                src_zero = int((src == 0).sum().item())
                                dst_zero = int((dst == 0).sum().item())
                                src_numel = int(src.numel())
                                dst_numel = int(dst.numel())
                            except Exception as exc:  # pragma: no cover - debug only
                                logger.warning(
                                    "[SpecAgent][LMC-H2D-%s] failed layer=%d "
                                    "token=%d buf_idx=%d slot=%d block=%d "
                                    "block_offset=%d buf_shape=%s dst_shape=%s err=%r",
                                    debug_kind,
                                    layer_id,
                                    token,
                                    buf_idx,
                                    slot,
                                    block,
                                    block_offset,
                                    tuple(tmp_gpu_buffer_obj.tensor.shape),
                                    tuple(kvcaches[layer_id].shape),
                                    exc,
                                )
                            else:
                                logger.warning(
                                    "[SpecAgent][LMC-H2D-%s] layer=%d "
                                    "token=%d buf_idx=%d slot=%d block=%d "
                                    "block_offset=%d buf_shape=%s dst_shape=%s "
                                    "src_sum=%.6g dst_sum=%.6g max_delta=%.6g "
                                    "src_zero=%d/%d dst_zero=%d/%d",
                                    debug_kind,
                                    layer_id,
                                    token,
                                    buf_idx,
                                    slot,
                                    block,
                                    block_offset,
                                    tuple(tmp_gpu_buffer_obj.tensor.shape),
                                    tuple(kvcaches[layer_id].shape),
                                    src_sum,
                                    dst_sum,
                                    max_delta,
                                    src_zero,
                                    src_numel,
                                    dst_zero,
                                    dst_numel,
                                )
                        debug_count += 1
                if (not defer_layer_submit
                        and os.environ.get("SPECAGENT_SCRATCH_TAG_DEBUG")):
                    tag_store = getattr(
                        builtins, "_specagent_scratch_layer_tags", None)
                    if not isinstance(tag_store, dict):
                        tag_store = {}
                        setattr(
                            builtins, "_specagent_scratch_layer_tags",
                            tag_store,
                        )
                    kv = kvcaches[layer_id]
                    tag_store[int(kv[0].data_ptr())] = {
                        "loaded_layer": int(layer_id),
                        "slot": int(layer_id % max(1, len(kvcaches))),
                        "num_tokens": int(num_tokens),
                        "starts": [int(x) for x in starts],
                        "ends": [int(x) for x in ends],
                    }
                if prof_end is not None:
                    prof_end = _specagent_record_cuda_event(
                        self.load_stream, enable_timing=True
                    )
                if (
                    isinstance(specagent_layer_events, list)
                    and not defer_layer_submit
                ):
                    event = _specagent_record_cuda_event(self.load_stream)
                    if layer_id < len(specagent_layer_events):
                        specagent_layer_events[layer_id] = event
            if bw_profile and prof_start is not None and prof_end is not None:
                bw_profile_records.append(
                    (layer_id, prof_h2d_bytes, prof_gpu_copy_bytes,
                     prof_start, prof_end)
                )
        if _hb:
            logger.error(
                "[SpecAgent][HOST-BREAKDOWN] layers=%d chunks_per_layer=%d "
                "copy_ms=%.2f xfer_ms=%.2f total_ms=%.2f",
                self.num_layers, len(starts),
                _hb_copy * 1000, _hb_xfer * 1000,
                (_hb_copy + _hb_xfer) * 1000,
            )
        yield

        # synchronize the last layer.  SpecAgent passes per-layer events and
        # waits on those from the attention stream; a stream-wide wait here
        # serializes unrelated load-stream work and defeats layerwise overlap.
        if sync and not event_synchronized_load:
            current_stream.wait_stream(self.load_stream)
        if bw_profile and bw_profile_records:
            total_h2d = 0
            total_gpu_copy = 0
            total_ms = 0.0
            for (
                layer_id,
                h2d_bytes,
                gpu_copy_bytes,
                start_event,
                end_event,
            ) in bw_profile_records:
                end_event.synchronize()
                elapsed_ms = float(start_event.elapsed_time(end_event))
                total_h2d += h2d_bytes
                total_gpu_copy += gpu_copy_bytes
                total_ms += elapsed_ms
                if layer_id < 4 or layer_id == self.num_layers - 1:
                    logger.info(
                        "[SpecAgent][LMC-BW] layer=%d h2d_bytes=%d "
                        "gpu_copy_bytes=%d elapsed_ms=%.3f h2d_gbps=%.3f "
                        "combined_gbps=%.3f",
                        layer_id,
                        h2d_bytes,
                        gpu_copy_bytes,
                        elapsed_ms,
                        h2d_bytes / elapsed_ms / 1.0e6
                        if elapsed_ms > 0 else 0.0,
                        (h2d_bytes + gpu_copy_bytes) / elapsed_ms / 1.0e6
                        if elapsed_ms > 0 else 0.0,
                    )
            logger.info(
                "[SpecAgent][LMC-BW] summary layers=%d h2d_bytes=%d "
                "gpu_copy_bytes=%d elapsed_ms_sum=%.3f h2d_gbps=%.3f "
                "combined_gbps=%.3f",
                len(bw_profile_records),
                total_h2d,
                total_gpu_copy,
                total_ms,
                total_h2d / total_ms / 1.0e6 if total_ms > 0 else 0.0,
                (total_h2d + total_gpu_copy) / total_ms / 1.0e6
                if total_ms > 0 else 0.0,
            )

        # free the buffer memory
        if tmp_gpu_buffer_obj is not None:
            if not defer_layer_submit:
                release_tokens = int(tmp_gpu_buffer_obj.tensor.shape[0])
                tmp_gpu_buffer_obj.ref_count_down()
                self._specagent_log_staging_physical(
                    "free", "load", release_tokens,
                )

        logger.debug(f"Finished loading all {self.num_layers} layers.")
        yield

    @_lmcache_nvtx_annotate
    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """
        This function is a generator that moves the KV cache from the paged GPU
        memory to the memory objects. The first iteration will prepare some
        related metadata and initiate the transfer in the first layer. In each
        of the following iterations, it will first wait until the storing of
        previous layer finishes, and then initiate string the KV cache of the
        current layer one. The storing process of the KV cache is paged GPU
        memory -> GPU buffer -> memory objects. The last iteration simply waits
        for the last layer to finish.
        In total, this the generator will yield num_layers + 1 times.

        :param memory_objs: The memory objects to store the KV cache. The first
            dimension is the number of layers, and the second dimension is the
            number of memory objects (i.e., number of chunks) for each layer.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        sync: bool = kwargs["sync"]
        offset: int = int(kwargs.get("offset", 0) or 0)
        defer_tmp_buffer_free = bool(
            kwargs.get("specagent_defer_store_tmp_buffer_free", True)
        )
        current_layer_only = bool(
            kwargs.get("specagent_save_current_layer_only", False)
        )
        defer_layerwise_commit = bool(
            kwargs.get("specagent_defer_layerwise_store_commit", False)
        )

        self._lazy_initialize_buffer(self.kvcaches)

        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            local_start = start - offset
            local_end = end - offset
            if local_start < 0 or local_end > len(slot_mapping):
                raise ValueError(
                    "Invalid layerwise store range after offset adjustment: "
                    f"start={start} end={end} offset={offset} "
                    f"slot_mapping_len={len(slot_mapping)}"
                )
            slot_mapping_chunks.append(slot_mapping[local_start:local_end])

        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)

        tmp_gpu_buffer_obj: Optional[MemoryObj] = None
        if self.use_gpu:
            buffer_shape = self.get_shape(num_tokens)
            assert self.gpu_buffer_allocator is not None
            self._specagent_release_completed_store_buffers()
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, MemoryFormat.KV_T2D
            )
            self._specagent_log_staging_physical(
                "alloc", "store", num_tokens,
                success=tmp_gpu_buffer_obj is not None,
            )
            if tmp_gpu_buffer_obj is None:
                self._specagent_wait_one_store_buffer()
                tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                    buffer_shape, self.dtype, MemoryFormat.KV_T2D
                )
                self._specagent_log_staging_physical(
                    "alloc_retry", "store", num_tokens,
                    success=tmp_gpu_buffer_obj is not None,
                )
            if tmp_gpu_buffer_obj is None:
                # Transient exhaustion: concurrent verify fetches can briefly
                # hold the whole staging pool (observed 2x on 2026-07-26 at
                # 24-30 concurrent requests; the previous assert killed the
                # engine).  Buffers free asynchronously as load-stream events
                # complete, so a bounded retry resolves it without new memory.
                _alloc_deadline = time.monotonic() + 5.0
                while tmp_gpu_buffer_obj is None:
                    self._specagent_release_completed_store_buffers()
                    self.specagent_flush_deferred_layer(-1)  # reap done plans
                    tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                        buffer_shape, self.dtype, MemoryFormat.KV_T2D
                    )
                    if tmp_gpu_buffer_obj is not None:
                        break
                    if time.monotonic() > _alloc_deadline:
                        raise RuntimeError(
                            "GPU staging buffer exhausted for >5s "
                            f"(need shape={buffer_shape}); concurrent "
                            "verify fetches exceed the staging pool"
                        )
                    time.sleep(0.002)
            assert tmp_gpu_buffer_obj is not None, (
                "Failed to allocate GPU buffer in GPUConnector"
            )
            assert tmp_gpu_buffer_obj.tensor is not None

        current_stream = torch.cuda.current_stream()

        for layer_id in range(self.num_layers):
            memory_objs_layer = memory_objs[layer_id]
            # kvcaches -> gpu_buffer -> memobj
            with torch.cuda.stream(self.store_stream):
                self.store_stream.wait_stream(current_stream)
                if self.use_gpu:
                    lmc_ops.single_layer_kv_transfer(
                        tmp_gpu_buffer_obj.tensor,
                        self.kvcaches[layer_id],
                        slot_mapping_full,
                        lmc_ops.TransferDirection.D2H,
                        self.gpu_kv_format,
                        token_major=True,
                    )
                    buffer_offset = 0
                    for start, end, memory_obj in zip(
                        starts, ends, memory_objs_layer, strict=False
                    ):
                        local_start = start - offset
                        local_end = end - offset
                        if local_start < 0 or local_end > len(slot_mapping):
                            raise ValueError(
                                "Invalid layerwise store range after offset adjustment: "
                                f"start={start} end={end} offset={offset} "
                                f"slot_mapping_len={len(slot_mapping)}"
                            )
                        chunk_len = local_end - local_start
                        assert memory_obj.tensor is not None
                        if self.use_gpu:
                            memory_obj.tensor.copy_(
                                tmp_gpu_buffer_obj.tensor[
                                    buffer_offset : buffer_offset + chunk_len
                                ],
                                non_blocking=True,
                            )
                            buffer_offset += chunk_len
                        else:
                            lmc_ops.single_layer_kv_transfer(
                                memory_obj.tensor,
                                self.kvcaches[layer_id],
                                slot_mapping[local_start:local_end],
                                lmc_ops.TransferDirection.D2H,
                                self.gpu_kv_format,
                                token_major=True,
                        )
                    # Set metadata format
                    if self.use_mla:
                        memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

            yield
            if (
                not defer_layerwise_commit
                and (sync or current_layer_only)
                and not (
                defer_tmp_buffer_free
                and self.use_gpu
                and layer_id == self.num_layers - 1
                )
            ):
                self.store_stream.synchronize()
            logger.debug(f"Finished offloading layer {layer_id}")
            if current_layer_only:
                # Give callers a synchronization point for this layer without
                # advancing into the next layer. SpecAgent invokes layerwise
                # save from each model-layer hook, so reading layer N+1 here
                # would race before that layer has produced KV.
                yield

        # In the deferred SpecAgent path, every layer copy was queued on this
        # one stream after the corresponding model-layer dependency.  A single
        # completion here is sufficient before CacheEngine publishes any CPU
        # memory object, and lets all earlier D2H copies overlap model compute.
        if defer_layerwise_commit:
            self.store_stream.synchronize()

        # free the buffer memory
        if tmp_gpu_buffer_obj is not None:
            if defer_tmp_buffer_free:
                self._specagent_defer_store_buffer_free(tmp_gpu_buffer_obj)
            else:
                release_tokens = int(tmp_gpu_buffer_obj.tensor.shape[0])
                tmp_gpu_buffer_obj.ref_count_down()
                self._specagent_log_staging_physical(
                    "free", "store", release_tokens,
                )

        yield

    def get_shape(self, num_tokens: int) -> torch.Size:
        if self.use_mla:
            # MLA format: [num_tokens, hidden_dim_size]
            return torch.Size([num_tokens, self.hidden_dim_size])
        else:
            # Standard format: [num_tokens, 2, hidden_dim_size]
            return torch.Size([num_tokens, 2, self.hidden_dim_size])


class SGLangGPUConnector(GPUConnectorInterface):
    """
    The GPU KV cache should be a list of tensors, one for each layer,
    with separate key and value pointers.
    More specifically, we have:
    - kvcaches: Tuple[List[Tensor], List[Tensor]]
      - The first element is a list of key tensors, one per layer.
      - The second element is a list of value tensors, one per layer.
    - Each tensor: [page_buffer_size, head_num, head_size]

    The connector manages the transfer of KV cache data between CPU and GPU
    memory for SGLang using pointer arrays for efficient access.
    It will produce/consume memory objects with KV_2LTD format.
    """

    def __init__(
        self, hidden_dim_size: int, num_layers: int, use_gpu: bool = False, **kwargs
    ):
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers

        self.kv_cache_pointers_on_gpu: dict[int, torch.Tensor] = {}
        self.page_buffer_size = 0

        self.gpu_buffer: Optional[torch.Tensor] = None
        self.use_mla = "use_mla" in kwargs and kwargs["use_mla"]

        self.num_kv_cache = num_layers if self.use_mla else num_layers * 2
        self.kv_cache_pointers = torch.empty(
            self.num_kv_cache, dtype=torch.int64, device="cpu"
        )

        if use_gpu:
            assert "chunk_size" in kwargs, (
                "chunk_size should be provided to create a GPU buffer."
            )
            assert "device" in kwargs, (
                "device should be provided to create a GPU buffer."
            )
            shape = self.get_shape(kwargs["chunk_size"])
            self.gpu_buffer = torch.empty(
                shape, dtype=kwargs["dtype"], device=kwargs["device"]
            )
            logger.info(f"GPU buffer: {self.gpu_buffer.shape}")

    def _initialize_pointers(self, kv_caches: List[torch.Tensor]) -> torch.Tensor:
        # Discover format first to handle flattening correctly
        self.gpu_kv_format = discover_gpu_kv_format(kv_caches, EngineType.SGLANG)

        # For TWO_X_NL_X_NBBS_NH_HS format, kv_caches is [[k_list], [v_list]]
        # We need to flatten it to [k0, k1, ..., v0, v1, ...]
        if self.gpu_kv_format == lmc_ops.GPUKVFormat.TWO_X_NL_X_NBBS_NH_HS:
            flat_kv_caches = kv_caches[0] + kv_caches[1]  # [k_list] + [v_list]
            device = flat_kv_caches[0].device
        else:
            flat_kv_caches = kv_caches
            device = flat_kv_caches[0].device

        assert len(flat_kv_caches) == self.num_kv_cache, (
            f"Expected {self.num_kv_cache} KV caches, got {len(flat_kv_caches)}"
        )

        self.kv_cache_pointers.numpy()[:] = [t.data_ptr() for t in flat_kv_caches]
        assert device.type == "cuda", "The device should be CUDA."
        idx = device.index
        if idx not in self.kv_cache_pointers_on_gpu:
            self.kv_cache_pointers_on_gpu[idx] = torch.empty(
                self.num_kv_cache, dtype=torch.int64, device=device
            )
        self.kv_cache_pointers_on_gpu[idx].copy_(self.kv_cache_pointers)

        # sglang MLA kv_caches[0].shape: [num_pages * page_size, 1, head_size]
        # sglang MHA kv_caches: [[k_list], [v_list]]
        # each with shape [num_pages * page_size, num_heads, head_size]
        self.page_buffer_size = get_page_buffer_size(kv_caches, self.gpu_kv_format)
        return self.kv_cache_pointers_on_gpu[idx]

    @_lmcache_nvtx_annotate
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Note:
          1. This function expects the 'slot_mapping' is a "partial slot mapping"
             where its length is the same as the uncached token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)


        :raises ValueError: If 'kvcaches' is not provided in kwargs.
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        if self.use_mla:
            if memory_obj.metadata.fmt != MemoryFormat.KV_MLA_FMT:
                raise ValueError(
                    "The memory object should be in KV_MLA_FMT format in"
                    f" order to be processed by {self.__class__.__name__}"
                )
        else:
            if memory_obj.metadata.fmt != MemoryFormat.KV_2LTD:
                raise ValueError(
                    "The memory object should be in KV_2LTD format in"
                    f" order to be processed by {self.__class__.__name__}"
                )

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        offset = kwargs.get("offset", 0)

        kvcaches: List[torch.Tensor] = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        kv_cache_pointers = self._initialize_pointers(kvcaches)
        lmc_ops.multi_layer_kv_transfer_unilateral(
            memory_obj.tensor,
            kv_cache_pointers,
            slot_mapping[start - offset : end - offset],
            kvcaches[0][0].device,
            self.page_buffer_size,
            lmc_ops.TransferDirection.H2D,
            self.gpu_kv_format,
        )

    @_lmcache_nvtx_annotate
    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Will set the memory_obj.metadata.fmt to MemoryFormat.KV_2LTD.

        Note:
          1. This function expects the 'slot_mapping' is a "partial slot mapping"
             where its length is the same as the uncached token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)

        :raises ValueError: If 'kvcaches' is not provided in kwargs,
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        kvcaches: List[torch.Tensor] = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        kv_cache_pointers = self._initialize_pointers(kvcaches)

        if self.gpu_buffer is None or end - start != self.gpu_buffer.shape[2]:
            lmc_ops.multi_layer_kv_transfer_unilateral(
                memory_obj.tensor,
                kv_cache_pointers,
                slot_mapping[start:end],
                kvcaches[0][0].device,
                self.page_buffer_size,
                lmc_ops.TransferDirection.D2H,
                self.gpu_kv_format,
            )
        else:
            # kvcaches -> gpu_buffer -> memobj
            assert self.gpu_buffer.device == kvcaches[0][0].device
            tmp_gpu_buffer = self.gpu_buffer[:, :, : end - start, :]
            lmc_ops.multi_layer_kv_transfer_unilateral(
                tmp_gpu_buffer,
                kv_cache_pointers,
                slot_mapping[start:end],
                kvcaches[0][0].device,
                self.page_buffer_size,
                lmc_ops.TransferDirection.D2H,
                self.gpu_kv_format,
            )
            memory_obj.tensor.copy_(tmp_gpu_buffer, non_blocking=True)

        if not memory_obj.tensor.is_cuda:
            # Force a synchronize if the target buffer is NOT CUDA device
            # NOTE: for better performance, we may not want to sync for every
            # memory object
            torch.cuda.synchronize()

        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size([2, self.num_layers, num_tokens, self.hidden_dim_size])

    # TODO(Jiayi): need to optimize to enable real batching
    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.to_gpu(memory_obj, start, end, **kwargs)

    # TODO(Yuwei): need to optimize to enable real batching
    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.from_gpu(memory_obj, start, end, **kwargs)


# TODO: support MLA
class SGLangLayerwiseGPUConnector(GPUConnectorInterface):
    """
    The GPU KV cache should be a list of tensors, one for each layer,
    with separate key and value pointers.
    More specifically, we have:
    - kvcaches: Tuple[List[Tensor], List[Tensor]]
      - The first element is a list of key tensors, one per layer.
      - The second element is a list of value tensors, one per layer.
    - Each tensor: [page_buffer_size, head_num, head_size]

    The connector manages the transfer of KV cache data between CPU and GPU
    memory for SGLang using pointer arrays for efficient access.
    It will produce/consume memory objects with KV_2LTD format.
    """

    def __init__(
        self, hidden_dim_size: int, num_layers: int, use_gpu: bool = False, **kwargs
    ):
        assert "dtype" in kwargs, "dtype should be provided to create a GPU buffer."
        self.dtype = kwargs["dtype"]
        assert "device" in kwargs, "device should be provided to create a GPU buffer."
        self.device = kwargs["device"]

        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers

        self.kv_cache_pointers_on_gpu: dict[int, torch.Tensor] = {}
        self.page_buffer_size = 0

        self.gpu_buffer: Optional[torch.Tensor] = None
        self.use_mla = "use_mla" in kwargs and kwargs["use_mla"]

        self.num_kv_cache = num_layers if self.use_mla else num_layers * 2
        self.element_size = torch.tensor([], dtype=self.dtype).element_size()
        self.kv_cache_pointers = torch.empty(
            self.num_kv_cache, dtype=torch.int64, device="cpu"
        )
        self.use_gpu = use_gpu
        self.gpu_buffer_allocator: Optional[GPUMemoryAllocator] = None

    def _lazy_initialize_buffer(self, kv_caches):
        """
        Lazily initialize the GPU buffer allocator if it is not initialized yet.
        Currently, we use the `kv_caches` (kv cache pointer) to determine
        the gpu buffer size in gpu connector.
        Also, the first request might be a bit slower due to buffer creation.
        """
        if self.use_gpu and self.gpu_buffer_allocator is None:
            self.gpu_kv_format = discover_gpu_kv_format(kv_caches, EngineType.SGLANG)
            self.tokens_per_layer = get_tokens_per_layer(kv_caches, self.gpu_kv_format)
            self.elements_per_layer = get_elements_per_layer(
                kv_caches, self.gpu_kv_format
            )
            logger.info(
                f"Lazily initializing GPU buffer (max tokens={self.tokens_per_layer})."
            )
            gpu_buffer_size = self.elements_per_layer * self.element_size
            logger.info(
                f"Lazily initializing GPU buffer (gpu buffer size={gpu_buffer_size})."
            )
            self.gpu_buffer_allocator = GPUMemoryAllocator(
                gpu_buffer_size, device=self.device
            )

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        raise NotImplementedError

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        raise NotImplementedError

    @_lmcache_nvtx_annotate
    def batched_to_gpu(self, starts: List[int], ends: List[int], **kwargs):
        """
        This function is a generator that moves the KV cache from the memory
        objects to paged GPU memory. The first iteration will prepare some
        related metadata. In each of the following iterations, it will first
        wait until the loading of the previous layer finish, and then load
        one layer of KV cache from the memory objects -> GPU buffer ->
        paged GPU memory. The last iteration simply waits for the last layer
        to finish.
        In total, this the generator will yield num_layers + 2 times.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        self._lazy_initialize_buffer(self.kvcaches)

        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])

        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)

        if self.use_gpu:
            buffer_shape = self.get_shape(num_tokens)

            assert self.gpu_buffer_allocator is not None, (
                "GPU buffer allocator should be initialized"
            )
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, MemoryFormat.KV_T2D
            )
            assert tmp_gpu_buffer_obj is not None, (
                "Failed to allocate GPU buffer in GPUConnector"
            )
            assert tmp_gpu_buffer_obj.tensor is not None

        offset = starts[0]

        for layer_id in range(self.num_layers):
            memory_objs_layer = yield
            if layer_id > 0:
                logger.debug(f"Finished loading layer {layer_id - 1}")

            # memobj -> gpu_buffer -> kvcaches
            for start, end, memory_obj in zip(
                starts, ends, memory_objs_layer, strict=False
            ):
                assert memory_obj.metadata.fmt == MemoryFormat.KV_T2D
                if self.use_gpu:
                    tmp_gpu_buffer_obj.tensor[start - offset : end - offset].copy_(
                        memory_obj.tensor, non_blocking=True
                    )
                else:
                    lmc_ops.single_layer_kv_transfer_sgl(
                        memory_obj.tensor,
                        self.kvcaches[0][layer_id],
                        self.kvcaches[1][layer_id],
                        slot_mapping[start:end],
                        lmc_ops.TransferDirection.H2D,
                        token_major=True,
                    )

            if self.use_gpu:
                t, h, d = self.kvcaches[0][layer_id].shape

                lmc_ops.single_layer_kv_transfer_sgl(
                    tmp_gpu_buffer_obj.tensor,
                    self.kvcaches[0][layer_id].view(t, 1, h, d),
                    self.kvcaches[1][layer_id].view(t, 1, h, d),
                    slot_mapping_full,
                    lmc_ops.TransferDirection.H2D,
                    token_major=True,
                )

        # free the buffer memory
        if self.use_gpu:
            tmp_gpu_buffer_obj.ref_count_down()

        logger.debug(f"Finished loading layer {layer_id}")
        yield

    @_lmcache_nvtx_annotate
    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """
        This function is a generator that moves the KV cache from the paged GPU
        memory to the memory objects. The first iteration will prepare some
        related metadata and initiate the transfer in the first layer. In each
        of the following iterations, it will first wait until the storing of
        previous layer finishes, and then initiate string the KV cache of the
        current layer one. The storing process of the KV cache is paged GPU
        memory -> GPU buffer -> memory objects. The last iteration simply waits
        for the last layer to finish.
        In total, this the generator will yield num_layers + 1 times.

        :param memory_objs: The memory objects to store the KV cache. The first
            dimension is the number of layers, and the second dimension is the
            number of memory objects (i.e., number of chunks) for each layer.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        self._lazy_initialize_buffer(self.kvcaches)

        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])

        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)

        if self.use_gpu:
            buffer_shape = self.get_shape(num_tokens)

            assert self.gpu_buffer_allocator is not None, (
                "GPU buffer allocator should be initialized"
            )
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, MemoryFormat.KV_T2D
            )
            assert tmp_gpu_buffer_obj is not None, (
                "Failed to allocate GPU buffer in GPUConnector"
            )
            assert tmp_gpu_buffer_obj.tensor is not None

        for layer_id in range(self.num_layers):
            memory_objs_layer = memory_objs[layer_id]
            # kvcaches -> gpu_buffer -> memobj
            if self.use_gpu:
                t, h, d = self.kvcaches[0][layer_id].shape
                lmc_ops.single_layer_kv_transfer_sgl(
                    tmp_gpu_buffer_obj.tensor,
                    self.kvcaches[0][layer_id].view(t, 1, h, d),
                    self.kvcaches[1][layer_id].view(t, 1, h, d),
                    slot_mapping_full,
                    lmc_ops.TransferDirection.D2H,
                    token_major=True,
                )

            start_idx = 0

            for start, end, memory_obj in zip(
                starts, ends, memory_objs_layer, strict=False
            ):
                assert memory_obj.tensor is not None
                if self.use_gpu:
                    chunk_len = memory_obj.tensor.shape[0]
                    memory_obj.tensor.copy_(
                        tmp_gpu_buffer_obj.tensor[start_idx : start_idx + chunk_len],
                        non_blocking=True,
                    )
                    start_idx += chunk_len
                else:
                    lmc_ops.single_layer_kv_transfer_sgl(
                        memory_obj.tensor,
                        self.kvcaches[0][layer_id],
                        self.kvcaches[1][layer_id],
                        slot_mapping[start:end],
                        lmc_ops.TransferDirection.D2H,
                        token_major=True,
                    )

            yield
            logger.debug(f"Finished offloading layer {layer_id}")

        # free the buffer memory
        if self.use_gpu:
            tmp_gpu_buffer_obj.ref_count_down()
        yield

    def get_shape(self, num_tokens: int) -> torch.Size:
        # TODO: support MLA
        return torch.Size([num_tokens, 2, self.hidden_dim_size])
