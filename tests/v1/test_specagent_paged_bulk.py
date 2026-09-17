# SPDX-License-Identifier: Apache-2.0
"""GPU coverage and lease-lifetime checks for paged bulk restoration."""
from types import SimpleNamespace as NS

import pytest
import torch

from lmcache.v1.specagent_paged_bulk import restore_paged_bulk


@pytest.mark.parametrize("fail", [False, True])
def test_leases_released_after_connector(fail):
    events = []
    obj = NS(ref_count_down=lambda: events.append("release"))
    segment = NS(starts=[0], ends=[16], memory_objs_by_layer=[[obj]])
    prepared = NS(claim=lambda index: segment,
                  release_unclaimed=lambda: events.append("unclaimed"))

    def transfer(*args):
        assert not events
        events.append("completed")
        if fail:
            raise RuntimeError("submission failed")

    engine = NS(is_healthy=lambda: True, num_layers=1,
                plan_sparse_layerwise_chunks=lambda *args: [[(0, 16, None)]],
                prepare_sparse_layerwise_plans=lambda plans: prepared,
                gpu_connector=NS(specagent_restore_paged_bulk=transfer))
    if fail:
        with pytest.raises(RuntimeError, match="submission failed"):
            restore_paged_bulk(engine, list(range(16)), [(0, 16)], [(3, 15)], [], None)
    else:
        assert restore_paged_bulk(engine, list(range(16)), [(0, 16)], [(3, 15)], [], None) == 12
    assert events == ["completed", "release", "unclaimed"]


@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("chunks", [3, 257])
def test_gpu_exact_write_ranges(layout, chunks):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemLayerwiseGPUConnector

    layers, block, heads, dim = 3, 16, 2, 8
    count = chunks * block
    connector = VLLMPagedMemLayerwiseGPUConnector(
        heads * dim, layers, dtype=torch.bfloat16, device=torch.device("cuda"),
        chunk_size=block, layout_hints={"kv_layout": layout})
    caches = []
    for _ in range(layers):
        shape = ((2, chunks + 1, block, heads, dim) if layout == "NHD"
                 else (2, chunks + 1, heads, block, dim))
        physical = torch.full(shape, -7, dtype=torch.bfloat16, device="cuda")
        caches.append(physical if layout == "NHD" else physical.permute(0, 1, 3, 2, 4))
    cpu = [torch.randn(count, 2, heads * dim, dtype=torch.bfloat16).pin_memory()
           for _ in range(layers)]
    sources = [[NS(tensor=source[i:i + block]) for i in range(0, count, block)]
               for source in cpu]
    slots = torch.arange(count, dtype=torch.int64, device="cuda")
    ranges = [(3, 15), (19, count - 2)]
    connector.specagent_restore_paged_bulk(
        sources, list(range(0, count, block)), list(range(block, count + 1, block)),
        ranges, caches, slots)
    for source, cache in zip(cpu, caches):
        expected = torch.full((2, (chunks + 1) * block, heads, dim), -7,
                              dtype=torch.bfloat16)
        for start, end in ranges:
            expected[:, start:end] = source[start:end].reshape(-1, 2, heads, dim).permute(1, 0, 2, 3)
        torch.testing.assert_close(cache.cpu().reshape_as(expected), expected, rtol=0, atol=0)
