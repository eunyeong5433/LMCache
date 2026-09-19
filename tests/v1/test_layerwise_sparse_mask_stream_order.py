# SPDX-License-Identifier: Apache-2.0
"""A layerwise load must not consume an unfinished destination mapping."""

# Standard
from types import SimpleNamespace

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.gpu_connectors import (
    VLLMPagedMemLayerwiseGPUConnector,
)


@pytest.mark.parametrize("mode", ["none", "all", "blocks", "alternating", "edges", "empty"])
def test_sparse_mask_values_and_stream_order(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    monkeypatch.setenv("SPECAGENT_DIRECT_HOST_SCATTER", "1")
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    mapping = torch.arange(256, dtype=torch.int64, device="cuda")
    indices = torch.arange(256)
    mask = torch.ones(256, dtype=torch.bool)
    if mode == "blocks":
        mask = (indices // 16) % 3 != 0
    elif mode == "alternating":
        mask = indices % 2 == 0
    elif mode == "edges":
        mask = (indices < 17) | (indices >= 239)
    elif mode == "empty":
        mask[:] = False
    expected = indices if mode == "none" else indices[mask]
    produced = torch.full((len(expected),), -777, device="cuda", dtype=torch.int64)
    observed = torch.empty_like(produced)
    target = torch.empty((2, 32, 16, 1, 8), dtype=torch.bfloat16, device="cuda")
    torch.cuda.synchronize()

    # Isolate the public generator's metadata preparation from allocation and
    # cache lookup. No scatter reads intentionally invalid addresses.
    connector = SimpleNamespace(
        kvcaches=[target],
        use_gpu=True,
        use_mla=False,
        num_layers=1,
        device=torch.device("cuda"),
        load_stream=consumer,
        initialize_kvcaches_ptr=lambda **kwargs: None,
        _specagent_release_completed_store_buffers=lambda: None,
        _lazy_initialize_buffer=lambda *args, **kwargs: None,
        _specagent_claim_pending_fetch_reservation=lambda *args: None,
        get_shape=lambda count: torch.Size([count, 1]),
        dtype=torch.bfloat16,
        gpu_buffer_allocator=SimpleNamespace(allocate=lambda *args: SimpleNamespace(
            tensor=torch.empty((0, 1), device="cuda"), ref_count_down=lambda: None,
        )),
        _specagent_log_staging_physical=lambda *args, **kwargs: None,
    )
    original_cat = torch.cat

    def controlled_cat(tensors: list, dim: int = 0, **kwargs) -> torch.Tensor:
        return original_cat(tensors, dim=dim, out=produced)

    generator = None
    try:
        with torch.cuda.stream(producer):
            old_reuse_event = torch.cuda.Event()
            old_reuse_event.record()
            torch.cuda._sleep(500_000_000)
            with monkeypatch.context() as patch:
                patch.setattr(torch, "cat", controlled_cat)
                generator = VLLMPagedMemLayerwiseGPUConnector.batched_to_gpu(
                    connector,
                    [0, 64, 128],
                    [64, 128, 256],
                    slot_mapping=mapping,
                    sync=True,
                    specagent_layer_events=[None],
                    specagent_defer_layer_submit=True,
                    specagent_write_token_mask=(
                        mask if mode != "none" else None
                    ),
                )
                next(generator)
        with torch.cuda.stream(consumer):
            consumer.wait_event(old_reuse_event)
            observed.copy_(produced)
        consumer.synchronize()
        torch.testing.assert_close(observed.cpu(), expected)
    finally:
        producer.synchronize()
        consumer.synchronize()
        if generator is not None:
            generator.close()
