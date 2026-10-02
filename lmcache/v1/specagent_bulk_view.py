# SPDX-License-Identifier: Apache-2.0
"""Leased, zero-copy T2D views of native 2LTD CPU cache chunks."""

from dataclasses import dataclass
from typing import Optional
import torch
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)


EMPTY_VIEW_RAW = torch.empty(0, dtype=torch.uint8)


def _validated_owner_tensor(owner: MemoryObj) -> torch.Tensor:
    """The layout checks of _validated_owner_views, without building views."""
    tensor = owner.tensor
    if (
        not owner.is_valid()
        or tensor is None
        or owner.get_memory_format() != MemoryFormat.KV_2LTD
        or tensor.device.type != "cpu"
        or tensor.ndim != 4
        or tensor.shape[0] != 2
        or not tensor.is_contiguous()
    ):
        raise ValueError("Expected a live contiguous CPU 2LTD chunk")
    return tensor


def _borrowed_descriptor(view: torch.Tensor) -> "BorrowedBulkLayerView":
    return BorrowedBulkLayerView(
        view,
        MemoryObjMetadata(
            shape=view.shape,
            dtype=view.dtype,
            address=view.data_ptr(),
            phy_size=0,
            ref_count=0,
            fmt=MemoryFormat.KV_T2D,
        ),
    )


class _LazyOwnerDescriptors:
    """SPECAGENT_LAZY_BULK_VIEWS=1: the per-owner descriptor tuple of
    borrow_bulk_layer_views, built per layer on first access and then reused
    (same object each time). Layer l is tensor.select(1, l).permute(1, 0, 2),
    identical to tensor.unbind(1)[l].permute(1, 0, 2)."""

    __slots__ = ("tensor", "items")

    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor
        self.items: list = [None] * int(tensor.shape[1])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, layer):
        if isinstance(layer, slice):
            return tuple(self[i] for i in range(*layer.indices(len(self))))
        item = self.items[layer]
        if item is None:
            if layer < 0:
                layer += len(self.items)
            item = _borrowed_descriptor(self.tensor.select(1, layer).permute(1, 0, 2))
            self.items[layer] = item
        return item

    def __iter__(self):
        for layer in range(len(self.items)):
            yield self[layer]


def _validated_owner_views(owner: MemoryObj) -> tuple[torch.Tensor, ...]:
    tensor = owner.tensor
    if (
        not owner.is_valid()
        or tensor is None
        or owner.get_memory_format() != MemoryFormat.KV_2LTD
        or tensor.device.type != "cpu"
        or tensor.ndim != 4
        or tensor.shape[0] != 2
        or not tensor.is_contiguous()
    ):
        raise ValueError("Expected a live contiguous CPU 2LTD chunk")
    cached = getattr(owner, "specagent_bulk_layer_views", None)
    if cached is None or cached[0] is not tensor:
        # Only tensor metadata is cached, never an acquired reader.
        cached = (tensor, tuple(row.permute(1, 0, 2) for row in tensor.unbind(1)))
        owner.specagent_bulk_layer_views = cached
    return cached[1]


class BulkLayerViewRelease:
    """Keep a bulk allocation alive until its last layer reader finishes."""

    def free(self, view: "BulkLayerView", allocator_type: Optional[str] = None) -> None:
        if view.is_valid():
            view.invalidate()
            view.owner.ref_count_down()


class BulkLayerView(TensorMemoryObj):
    """A strided T2D view owning one reference to a native CPU bulk chunk.

    The caller must already own a live reference while constructing this
    reader. The view takes an additional independent reference; its lifetime
    therefore survives removal of the chunk from the CPU cache.
    """

    def __init__(self, owner: MemoryObj, layer: int) -> None:
        self.parent_allocator = None
        self.valid = False
        views = _validated_owner_views(owner)
        if not 0 <= layer < len(views):
            raise IndexError("Layer index outside native bulk chunk")
        self._initialize(owner, views[layer])

    @classmethod
    def acquire_layers(cls, owner: MemoryObj, num_layers: int) -> list["BulkLayerView"]:
        """Acquire independent readers after validating their shared owner once.

        The caller must hold an owner reference for the whole acquisition.
        Each returned reader retains its own owner reference. On failure,
        release every reader acquired here before propagating the exception.
        """
        tensors = _validated_owner_views(owner)
        if len(tensors) != num_layers:
            raise ValueError("Native bulk owner has an incomplete layer count")
        readers: list[BulkLayerView] = []
        try:
            for tensor in tensors:
                reader = cls.__new__(cls)
                reader._initialize(owner, tensor)
                readers.append(reader)
        except BaseException:
            for reader in readers:
                reader.ref_count_down()
            raise
        return readers

    def _initialize(self, owner: MemoryObj, view: torch.Tensor) -> None:
        self.parent_allocator = None
        self.valid = False
        meta = MemoryObjMetadata(
            shape=view.shape,
            dtype=view.dtype,
            address=view.data_ptr(),
            phy_size=0,
            ref_count=1,
            fmt=MemoryFormat.KV_T2D,
        )
        owner.ref_count_up()
        try:
            self.owner = owner
            self.layer_tensor = view
            super().__init__(EMPTY_VIEW_RAW, meta, BulkLayerViewRelease())
        except BaseException:
            self.parent_allocator = None
            self.valid = False
            owner.ref_count_down()
            raise

    @property
    def tensor(self) -> Optional[torch.Tensor]:
        return self.layer_tensor if self.is_valid() else None

    @property
    def data_ptr(self) -> int:
        if not self.is_valid():
            raise RuntimeError("Layer view has been released")
        return self.layer_tensor.data_ptr()

    @property
    def raw_tensor(self) -> torch.Tensor:
        raise RuntimeError("A strided layer view cannot be serialized as raw bytes")

    @property
    def byte_array(self) -> bytes:
        raise RuntimeError("A strided layer view is local-CPU only")


def acquire_bulk_layer_views(
    owners: list[MemoryObj], num_layers: int
) -> list[list[MemoryObj]]:
    """Lease all layer views; caller retains responsibility for its owner leases."""
    layers: list[list[MemoryObj]] = [[] for _ in range(num_layers)]
    try:
        for owner in owners:
            readers = BulkLayerView.acquire_layers(owner, num_layers)
            for layer, reader in zip(layers, readers, strict=True):
                layer.append(reader)
    except BaseException:
        for layer in layers:
            for view in layer:
                view.ref_count_down()
        raise
    return layers


@dataclass(frozen=True, slots=True)
class BorrowedBulkLayerView:
    """Read-only tensor descriptor; the containing fetch plan owns the CPU lease."""

    tensor: torch.Tensor
    metadata: MemoryObjMetadata


class _BorrowedLayerColumn:
    """Lazy column ``[chunk[layer] for chunk in chunks[start:stop]]``.

    The per-owner descriptor tuples are immutable, so a column is a stable
    read-only sequence. It supports the list operations its consumers use:
    len, integer and slice indexing, and iteration.
    """

    specagent_layer_column = True
    __slots__ = ("chunks", "layer", "start", "stop")

    def __init__(self, chunks, layer: int, start: int = 0, stop: Optional[int] = None):
        self.chunks = chunks
        self.layer = layer
        self.start = start
        self.stop = len(chunks) if stop is None else stop

    def __len__(self) -> int:
        return self.stop - self.start

    def __iter__(self):
        layer = self.layer
        chunks = self.chunks
        for index in range(self.start, self.stop):
            yield chunks[index][layer]

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step != 1:
                return list(self)[index]
            return _BorrowedLayerColumn(
                self.chunks, self.layer, self.start + start, self.start + stop
            )
        n = len(self)
        if index < 0:
            index += n
        if not 0 <= index < n:
            raise IndexError("layer column index out of range")
        return self.chunks[self.start + index][self.layer]


def borrow_bulk_layer_views(
    owners: list[MemoryObj], num_layers: int
) -> list[list[BorrowedBulkLayerView]]:
    """Reuse tensor descriptors while the caller retains every owner lease.

    No reference is acquired or released here. Descriptors must only be sent
    to a deferred fetch consumer that retains the owners until GPU completion.
    Returns one lazy column per layer (see _BorrowedLayerColumn).
    """
    import os as _os

    verify = _os.environ.get("SPECAGENT_FASTPATH_VERIFY") == "1"
    lazy = _os.environ.get("SPECAGENT_LAZY_BULK_VIEWS") == "1"
    chunks: list[tuple[BorrowedBulkLayerView, ...]] = []
    for owner in owners:
        cached = getattr(owner, "specagent_bulk_borrowed_views", None)
        if (
            cached is not None
            and len(cached) == 3
            and cached[2] is owner.tensor
            and len(cached[1]) == num_layers
            and owner.is_valid()
        ):
            # Same live tensor object: its layout checks cannot change.
            if verify:
                fresh = _validated_owner_views(owner)
                if len(fresh) != num_layers or any(
                    f.data_ptr() != d.tensor.data_ptr()
                    or tuple(f.shape) != tuple(d.tensor.shape)
                    or tuple(f.stride()) != tuple(d.tensor.stride())
                    for f, d in zip(fresh, cached[1], strict=True)
                ):
                    raise AssertionError("fastpath borrowed view cache mismatch")
            chunks.append(cached[1])
            continue
        if lazy:
            tensor = _validated_owner_tensor(owner)
            if tensor.shape[1] != num_layers:
                raise ValueError("Native bulk owner has an incomplete layer count")
            cached = (None, _LazyOwnerDescriptors(tensor), owner.tensor)
            if _os.environ.get("SPECAGENT_LAZY_BULK_VIEWS_CHECK") == "1":
                eager = _validated_owner_views(owner)
                if len(eager) != len(cached[1]) or any(
                    f.data_ptr() != d.tensor.data_ptr()
                    or tuple(f.shape) != tuple(d.tensor.shape)
                    or tuple(f.stride()) != tuple(d.tensor.stride())
                    or d.metadata.address != f.data_ptr()
                    or tuple(d.metadata.shape) != tuple(f.shape)
                    for f, d in zip(eager, cached[1], strict=True)
                ):
                    raise AssertionError("lazy bulk view descriptor mismatch")
            owner.specagent_bulk_borrowed_views = cached
            chunks.append(cached[1])
            continue
        tensors = _validated_owner_views(owner)
        if len(tensors) != num_layers:
            raise ValueError("Native bulk owner has an incomplete layer count")
        if cached is None or cached[0] is not tensors:
            descriptors = tuple(
                BorrowedBulkLayerView(
                    tensor,
                    MemoryObjMetadata(
                        shape=tensor.shape,
                        dtype=tensor.dtype,
                        address=tensor.data_ptr(),
                        phy_size=0,
                        ref_count=0,
                        fmt=MemoryFormat.KV_T2D,
                    ),
                )
                for tensor in tensors
            )
            cached = (tensors, descriptors, owner.tensor)
            owner.specagent_bulk_borrowed_views = cached
        elif len(cached) != 3:
            cached = (cached[0], cached[1], owner.tensor)
            owner.specagent_bulk_borrowed_views = cached
        chunks.append(cached[1])
    if not chunks:
        return [[] for _ in range(num_layers)]
    columns = [_BorrowedLayerColumn(chunks, layer) for layer in range(num_layers)]
    if verify:
        eager = [list(layer) for layer in zip(*chunks, strict=True)]
        if [list(column) for column in columns] != eager:
            raise AssertionError("fastpath borrowed layer columns mismatch")
    return columns


def merge_native_bulk_prefix(source: MemoryObj, target: MemoryObj, count: int) -> None:
    """Restore a saved partial CPU prefix after copying the new GPU suffix."""
    src, dst = source.tensor, target.tensor
    if (
        src is None
        or dst is None
        or src.device.type != "cpu"
        or dst.device.type != "cpu"
        or source.get_memory_format() != MemoryFormat.KV_2LTD
        or target.get_memory_format() != MemoryFormat.KV_2LTD
        or src.ndim != 4
        or dst.ndim != 4
        or src.shape[:2] != dst.shape[:2]
        or src.shape[3] != dst.shape[3]
        or src.dtype != dst.dtype
        or src.shape[2] != count
        or not 0 < count <= dst.shape[2]
    ):
        raise ValueError("Native bulk prefix merge shape/coverage mismatch")
    dst[:, :, :count, :].copy_(src)
