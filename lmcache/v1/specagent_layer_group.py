# SPDX-License-Identifier: Apache-2.0
"""CPU-only ownership of a complete chunk, retaining its layer tensors."""

# Standard
from functools import lru_cache
from typing import Optional

# Third Party
import torch

# First Party
from lmcache.utils import CacheEngineKey, LayerCacheEngineKey
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObj,
    MemoryObjMetadata,
    TensorMemoryObj,
)


@lru_cache(maxsize=8192)
def chunk_key(key: CacheEngineKey) -> CacheEngineKey:
    """Return the chunk owner key for a layer alias, preserving request tags."""
    if not isinstance(key, LayerCacheEngineKey):
        return key
    return CacheEngineKey(
        key.model_name,
        key.world_size,
        key.worker_id,
        key.chunk_hash,
        key.dtype,
        key.request_configs,
    )


class LayerGroupRelease:
    """Release the group's existing child leases; never free raw memory twice."""

    def free(self, group: "LayerGroup", allocator_type: Optional[str] = None) -> None:
        if not group.is_valid():
            return
        group.invalidate()
        for layer in group.layers:
            layer.ref_count_down()


class LayerGroup(TensorMemoryObj):
    """One local-CPU cache owner for all completed KV layers of a chunk.

    Consumes one existing reference per layer. A layer read acquires its own
    reference, so evicting this cache owner cannot recycle an in-flight read.
    The inherited pin/timeout logic applies to the complete chunk, once.
    No concatenation, CPU repack, or change to the layer's T2D bytes occurs.
    Groups must never be sent to a remote backend or a GPU connector directly.
    """

    def __init__(self, layers: list[MemoryObj]) -> None:
        if not layers or any(
            x.get_memory_format() != MemoryFormat.KV_T2D for x in layers
        ):
            raise ValueError("LayerGroup requires completed T2D layer objects")
        shape = layers[0].get_shape()
        if any(
            x.get_shape() != shape or x.get_dtype() != layers[0].get_dtype()
            for x in layers
        ):
            raise ValueError("LayerGroup layers must have identical shape and dtype")
        size = sum(x.get_size() for x in layers)
        meta = MemoryObjMetadata(
            shape=torch.Size([size]),
            dtype=torch.uint8,
            address=id(self),
            phy_size=sum(x.get_physical_size() for x in layers),
            ref_count=1,
            fmt=MemoryFormat.UNDEFINED,
        )
        self.layers = tuple(layers)
        super().__init__(torch.empty(0, dtype=torch.uint8), meta, LayerGroupRelease())

    def acquire_layer(self, index: int) -> MemoryObj:
        """Return a separately leased existing T2D object for a layer reader."""
        with self.lock:
            if not self.is_valid():
                raise RuntimeError("Cannot read an evicted layer group")
            layer = self.layers[index]
            layer.ref_count_up()
            return layer

    def acquire_layers(self, count: int) -> tuple[MemoryObj, ...]:
        """Lease every existing layer independently while holding one owner lock.

        Raises RuntimeError for an evicted or incomplete group. The caller
        must release one reference on each returned layer after consumption.
        """
        with self.lock:
            if not self.is_valid() or len(self.layers) != count:
                raise RuntimeError("Cannot acquire an incomplete layer group")
            acquired = []
            try:
                for layer in self.layers:
                    layer.ref_count_up()
                    acquired.append(layer)
            except BaseException:
                for layer in acquired:
                    layer.ref_count_down()
                raise
            return self.layers

    def get_num_tokens(self) -> int:
        return self.layers[0].get_shape()[0]

    @property
    def tensor(self) -> Optional[torch.Tensor]:
        raise RuntimeError("LayerGroup is an owner; read an individual layer")

    @property
    def raw_tensor(self) -> Optional[torch.Tensor]:
        raise RuntimeError("LayerGroup is not a serialized tensor")

    @property
    def data_ptr(self) -> int:
        raise RuntimeError("LayerGroup has no contiguous data pointer")

    @property
    def byte_array(self) -> bytes:
        raise RuntimeError("LayerGroup is local-CPU only")
