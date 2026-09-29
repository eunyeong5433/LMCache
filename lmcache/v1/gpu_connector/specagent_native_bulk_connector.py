# SPDX-License-Identifier: Apache-2.0
"""Native bulk NORMAL transfers with the existing SPEC layer fetch engine."""

# Standard
from typing import Any, Optional

# Third Party
import torch

# First Party
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.gpu_connector.utils import LayoutHints
from lmcache.v1.gpu_connector.gpu_connectors import (
    VLLMPagedMemGPUConnectorV2,
    VLLMPagedMemLayerwiseGPUConnector,
)


class SpecAgentNativeBulkGPUConnector(VLLMPagedMemLayerwiseGPUConnector):
    """Preserve native 2LTD bulk transfers; use layer views only during SPEC.

    Layerwise reads are initialized lazily by the inherited fetch engine.
    The NORMAL store/restore methods delegate directly to the same native
    connector used by OFF, including its completion/stream ordering.
    """

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheMetadata,
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
        layout_hints: Optional[LayoutHints] = None,
    ) -> "SpecAgentNativeBulkGPUConnector":
        """Construct native NORMAL and deferred layerwise SPEC transfer engines."""
        if metadata.use_mla:
            raise ValueError("Native bulk SPEC views do not support MLA")
        obj = super().from_metadata(metadata, use_gpu, device, layout_hints)
        obj.bulk_connector = VLLMPagedMemGPUConnectorV2.from_metadata(
            metadata, use_gpu, device, layout_hints
        )
        return obj

    def batched_to_gpu(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate bulk objects to NORMAL, or layer plans to SPEC retrieval."""
        if len(args) == 3:
            return self.bulk_connector.batched_to_gpu(*args, **kwargs)
        if len(args) == 2:
            return super().batched_to_gpu(*args, **kwargs)
        raise TypeError(
            "Expected bulk(objects, starts, ends) or layerwise(starts, ends)"
        )

    def batched_from_gpu(
        self,
        memory_objs: list[MemoryObj],
        starts: list[int],
        ends: list[int],
        **kwargs: Any,
    ) -> None:
        """Store native bulk owners after model production completes."""
        if memory_objs and isinstance(memory_objs[0], (list, tuple)):
            raise ValueError("Native bulk storage requires one owner per chunk")
        slot_mapping = kwargs.get("slot_mapping")
        if isinstance(slot_mapping, torch.Tensor) and slot_mapping.device.type == "cpu":
            # Keep the original blocking producer wait, after the engine has
            # prepared allocation keys and CPU proof hashes. Publication still
            # happens only after the synchronous bulk store completes.
            kvcaches = kwargs.get("kvcaches") or self.bulk_connector.kvcaches
            if not kvcaches:
                raise ValueError("kvcaches are required for CPU slot mappings")
            kwargs["slot_mapping"] = slot_mapping.to(kvcaches[0].device)
        return self.bulk_connector.batched_from_gpu(memory_objs, starts, ends, **kwargs)

    def specagent_restore_native_sparse(
        self,
        owners: list[MemoryObj],
        starts: list[int],
        ends: list[int],
        write_ranges: list[tuple[int, int]],
        kvcaches: list[torch.Tensor],
        slot_mapping: torch.Tensor,
    ) -> None:
        """Restore only missing paged slots, waiting before owner leases release.

        Chunk lookup remains chunk-aligned. Negative destination slots mask
        resident tokens independently of lookup, including a partial tail.
        """
        masked_slots = torch.full_like(slot_mapping, -1)
        for start, end in write_ranges:
            masked_slots[start:end].copy_(slot_mapping[start:end])
        stream = self.bulk_connector.load_stream
        stream.wait_stream(torch.cuda.current_stream(slot_mapping.device))
        try:
            self.bulk_connector.batched_to_gpu(
                owners,
                starts,
                ends,
                kvcaches=kvcaches,
                slot_mapping=masked_slots,
            )
        finally:
            # A later chunk can fail after an earlier async copy was submitted.
            stream.synchronize()
