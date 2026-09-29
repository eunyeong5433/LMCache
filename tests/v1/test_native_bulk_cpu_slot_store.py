# SPDX-License-Identifier: Apache-2.0
"""Native bulk accepts CPU slots and returns only after producer data is saved."""

# Standard
from types import SimpleNamespace

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemGPUConnectorV2
from lmcache.v1.gpu_connector.specagent_native_bulk_connector import (
    SpecAgentNativeBulkGPUConnector,
)
from lmcache.v1.memory_management import MemoryFormat


@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_cpu_slots_preserve_pending_producer_and_partial_tail(layout: str) -> None:
    """Pending GPU writes must be visible in every stored layer and tail."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    layers, heads, dim, block = 3, 2, 128, 16
    connector = object.__new__(SpecAgentNativeBulkGPUConnector)
    connector.bulk_connector = VLLMPagedMemGPUConnectorV2(
        heads * dim, layers, device=torch.device("cuda"), use_gpu=False,
        layout_hints={"kv_layout": layout},
    )
    caches = []
    for _ in range(layers):
        shape = ((2, 20, block, heads, dim) if layout == "NHD"
                 else (2, 20, heads, block, dim))
        physical = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
        caches.append(physical if layout == "NHD"
                      else physical.permute(0, 1, 3, 2, 4))
    slots = torch.arange(17, 290, dtype=torch.int64)
    tensors = [torch.empty((2, layers, n, heads * dim),
                           dtype=torch.bfloat16, pin_memory=True)
               for n in [256, 17]]
    owners = [SimpleNamespace(tensor=t, metadata=SimpleNamespace(
        fmt=MemoryFormat.KV_2LTD)) for t in tensors]
    torch.cuda._sleep(1000000)
    for layer, cache in enumerate(caches):
        cache[0].fill_(layer + 1)
        cache[1].fill_(layer + 11)
    connector.batched_from_gpu(owners, [0, 256], [256, 273],
                               kvcaches=caches, slot_mapping=slots)
    for tensor in tensors:
        for layer in range(layers):
            assert torch.all(tensor[0, layer] == layer + 1)
            assert torch.all(tensor[1, layer] == layer + 11)
