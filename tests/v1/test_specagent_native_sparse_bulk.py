# SPDX-License-Identifier: Apache-2.0
"""Native sparse paged restore must never overwrite resident/hole slots."""

from types import SimpleNamespace as NS
import pytest
import torch
from lmcache.v1.specagent_paged_bulk import restore_paged_bulk


@pytest.mark.parametrize("missing", [False, True])
def test_native_sparse_owner_lifetime(missing):
    events = []
    owners = [NS(ref_count_down=lambda: events.append("release"))]
    result = owners + ([None] if missing else [])
    plans = [[(0, 16, "a"), (16, 17, "b")]] if missing else [[(0, 16, "a")]]

    def copy(*args):
        assert not events
        events.append("completed")

    e = NS(
        native_bulk_spec=True,
        is_healthy=lambda: True,
        plan_sparse_layerwise_chunks=lambda *args: plans,
        storage_manager=NS(batched_get=lambda *args, **kwargs: result),
        gpu_connector=NS(specagent_restore_native_sparse=copy),
    )
    if missing:
        with pytest.raises(RuntimeError, match="incomplete"):
            restore_paged_bulk(e, list(range(17)), [(0, 17)], [(3, 9)], [], None)
        assert events == ["release"]
    else:
        assert (
            restore_paged_bulk(e, list(range(16)), [(0, 16)], [(3, 9)], [], None) == 6
        )
        assert events == ["completed", "release"]


@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_native_sparse_long_offset_partial_tail_exact(layout):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from lmcache.v1.gpu_connector.specagent_native_bulk_connector import (
        SpecAgentNativeBulkGPUConnector,
    )
    from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemGPUConnectorV2
    from lmcache.v1.memory_management import MemoryFormat

    layers, heads, dim, block = 3, 2, 128, 16
    connector = object.__new__(SpecAgentNativeBulkGPUConnector)
    connector.bulk_connector = VLLMPagedMemGPUConnectorV2(
        heads * dim,
        layers,
        device=torch.device("cuda"),
        use_gpu=False,
        layout_hints={"kv_layout": layout},
    )
    caches = []
    for _ in range(layers):
        shape = (
            (2, 20, block, heads, dim)
            if layout == "NHD"
            else (2, 20, heads, block, dim)
        )
        physical = torch.full(shape, -7, device="cuda", dtype=torch.bfloat16)
        caches.append(physical if layout == "NHD" else physical.permute(0, 1, 3, 2, 4))
    sources = [
        torch.randn((2, layers, n, heads * dim), dtype=torch.bfloat16).pin_memory()
        for n in [256, 17]
    ]
    owners = [NS(tensor=s, metadata=NS(fmt=MemoryFormat.KV_2LTD)) for s in sources]
    starts, ends = [65536, 65792], [65792, 65809]
    slots = torch.zeros(65809, device="cuda", dtype=torch.int64)
    slots[65536:] = torch.arange(16, 16 + 273, device="cuda")
    ranges = [(65539, 65549), (65800, 65805)]
    connector.specagent_restore_native_sparse(
        owners, starts, ends, ranges, caches, slots
    )
    for layer, cache in enumerate(caches):
        expected = torch.full((2, 320, heads, dim), -7, dtype=torch.bfloat16)
        for left, right in ranges:
            for token in range(left, right):
                index = 0 if token < starts[1] else 1
                dest = token - 65536 + 16
                expected[:, dest] = sources[index][
                    :, layer, token - starts[index]
                ].reshape(2, heads, dim)
        torch.testing.assert_close(
            cache.cpu().reshape_as(expected), expected, rtol=0, atol=0
        )
