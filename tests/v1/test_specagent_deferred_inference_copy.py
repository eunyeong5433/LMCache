# SPDX-License-Identifier: Apache-2.0
"""Deferred SPEC fetch must copy inference buffers from a producer thread."""

# Standard
from concurrent.futures import ThreadPoolExecutor

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemLayerwiseGPUConnector
from lmcache.v1.gpu_connector.utils import discover_gpu_kv_format, EngineType


@pytest.mark.parametrize("layout", ["NHD", "HND"])
def test_deferred_strided_inference_copy_from_background(layout: str) -> None:
    """A strided native layer view must arrive intact before the ready event."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    heads, dim, count = 2, 128, 17
    connector = VLLMPagedMemLayerwiseGPUConnector(
        heads * dim, 1, chunk_size=256, dtype=torch.bfloat16,
        device=torch.device("cuda"), layout_hints={"kv_layout": layout},
    )
    with torch.inference_mode():
        source = torch.randn((2, 3, count, heads * dim),
                             dtype=torch.bfloat16).pin_memory()
        layer = source[:, 1].permute(1, 0, 2)
        assert not layer.is_contiguous()
        transfer = torch.empty((count, 2, heads * dim),
                               dtype=torch.bfloat16, device="cuda")
        shape = ((2, 3, 16, heads, dim) if layout == "NHD"
                 else (2, 3, heads, 16, dim))
        physical = torch.full(shape, -7, dtype=torch.bfloat16, device="cuda")
        target = physical
        slots = torch.arange(7, 7 + count, dtype=torch.int64, device="cuda")
        slots[3] = -1
    events = [None]
    connector.gpu_kv_format = discover_gpu_kv_format(
        [target], EngineType.VLLM, layout_hints={"kv_layout": layout},
    )
    # Construct the deferred producer input at the connector boundary. These
    # fields are needed to exercise its public flush/event/lifetime contract.
    connector._specagent_deferred_layer_fetches[0] = [dict(
        copy_srcs=[layer], copy_dsts=[transfer], transfer_tensor=transfer,
        target_kv_cache=target, slot_mapping=slots, layer_events=events,
    )]
    torch.cuda.synchronize()

    def flush() -> bool:
        assert not torch.is_inference_mode_enabled()
        result = connector.specagent_flush_deferred_layer(0)
        assert not torch.is_inference_mode_enabled()
        return result

    with ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(flush).result(timeout=30)
    assert events[0] is not None
    events[0].synchronize()
    logical = target if layout == "NHD" else target.permute(0, 1, 3, 2, 4)
    actual = logical.cpu().reshape(2, 48, heads * dim)
    expected = torch.full_like(actual, -7)
    for i in range(count):
        if i != 3:
            expected[:, i + 7] = layer[i]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not connector.specagent_flush_deferred_layer(0)
