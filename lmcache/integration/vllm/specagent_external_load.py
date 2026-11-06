# SPDX-License-Identifier: Apache-2.0
"""Preserve an admitted paged load alongside a SpecAgent scratch fetch."""

# Standard
from dataclasses import dataclass
from typing import Any

# Third Party
import torch

# First Party
from lmcache.v1.specagent_paged_bulk import restore_paged_bulk


@dataclass
class SpecAgentExternalLoad:
    """One step's accepted external interval and its original CPU source key.

    ``token_ids`` ends at the connector lookup hit, which may include the last
    prompt token that vLLM recomputes. Only ``[start, end)`` is written. Keeping
    this metadata separate prevents scratch save trimming from changing the
    source chunk key, and avoids changing the request's CPU proof or save state.
    """

    req_id: str
    token_ids: list[int]
    block_ids: list[int]
    block_size: int
    start: int
    end: int
    request_configs: dict | None = None

    def restore(
        self,
        engine: Any,
        kvcaches: list[torch.Tensor],
        chunk_size: int,
        *,
        layerwise: bool,
    ) -> int:
        """Fill the paged interval before attention, or raise on missing KV.

        The existing bulk path waits for completion before releasing CPU
        leases, and uses no observation scratch or layerwise staging pages.
        The ordinary non-layerwise fallback also checks exact retrieval
        coverage. Resident prefix and new query slots are never written.
        """
        source_end = len(self.token_ids)
        if not (
            0 <= self.start < self.end <= source_end
            and self.block_size > 0
            and chunk_size > 0
            and source_end <= len(self.block_ids) * self.block_size
            and kvcaches
        ):
            raise RuntimeError(f"Invalid SpecAgent external load: {self.req_id}")
        first = self.start // self.block_size
        last = (self.end + self.block_size - 1) // self.block_size
        if any(b <= 0 for b in self.block_ids[first:last]):
            raise RuntimeError(
                f"SpecAgent external load has a null destination: {self.req_id}"
            )
        device = kvcaches[0].device
        blocks = torch.tensor(self.block_ids, dtype=torch.int64, device=device)
        positions = torch.arange(source_end, dtype=torch.int64, device=device)
        slots = blocks[positions // self.block_size] * self.block_size
        slots += positions % self.block_size
        if getattr(engine, "native_bulk_spec", False) or layerwise:
            return restore_paged_bulk(
                engine,
                self.token_ids,
                [(self.start // chunk_size * chunk_size, source_end)],
                [(self.start, self.end)],
                kvcaches,
                slots,
                self.request_configs,
            )
        # Ordinary retrieve requires a chunk-aligned False prefix followed
        # only by True. Mask destinations separately, including the final
        # prompt token that was looked up but will be recomputed.
        mask = torch.zeros(source_end, dtype=torch.bool)
        mask[self.start // chunk_size * chunk_size :] = True
        slots[: self.start] = -1
        slots[self.end :] = -1
        retrieved = engine.retrieve(
            self.token_ids,
            mask,
            kvcaches=kvcaches,
            slot_mapping=slots,
            vllm_cached_tokens=self.start,
            request_configs=self.request_configs,
            req_id=self.req_id,
        )
        if retrieved.numel() < self.end or not bool(
            retrieved[self.start : self.end].all()
        ):
            raise RuntimeError(
                f"SpecAgent external load is incomplete: {self.req_id} "
                f"range=[{self.start}, {self.end})"
            )
        return self.end - self.start
