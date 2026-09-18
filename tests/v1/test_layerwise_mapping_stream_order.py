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


@pytest.mark.parametrize("masked", [False, True])
def test_mapping_production_precedes_load(
    monkeypatch: pytest.MonkeyPatch, masked: bool
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    monkeypatch.setenv("SPECAGENT_DIRECT_HOST_SCATTER", "1")
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    mapping = torch.arange(256, dtype=torch.int64, device="cuda")
    produced = torch.full_like(mapping, -777)
    observed = torch.empty_like(mapping)
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
                    [0],
                    [256],
                    slot_mapping=mapping,
                    sync=True,
                    specagent_layer_events=[None],
                    specagent_defer_layer_submit=True,
                    specagent_write_token_mask=(
                        torch.ones(256, dtype=torch.bool) if masked else None
                    ),
                )
                next(generator)
        with torch.cuda.stream(consumer):
            consumer.wait_event(old_reuse_event)
            observed.copy_(produced)
        consumer.synchronize()
        torch.testing.assert_close(observed.cpu(), torch.arange(256))
    finally:
        producer.synchronize()
        consumer.synchronize()
        if generator is not None:
            generator.close()
