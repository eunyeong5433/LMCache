# SPDX-License-Identifier: Apache-2.0
"""SPECAGENT_B2G_FAST_SETUP helper == per-chunk slice/cat loop of batched_to_gpu."""
# Third Party
import numpy as np
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.gpu_connectors import _specagent_fast_slot_setup


def _reference(starts, ends, offset, slot_mapping):
    chunks = []
    for start, end in zip(starts, ends, strict=False):
        ls, le = start - offset, end - offset
        if ls < 0 or le > len(slot_mapping):
            raise ValueError(
                "Invalid layerwise store range after offset adjustment: "
                f"start={start} end={end} offset={offset} "
                f"slot_mapping_len={len(slot_mapping)}")
        chunks.append(slot_mapping[ls:le])
    return [len(c) for c in chunks], torch.cat(chunks, dim=0)


@pytest.mark.parametrize("seed", range(40))
def test_matches_reference(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(256, 20000))
    sm = torch.from_numpy(rng.integers(0, 1 << 30, size=n))
    offset = int(rng.integers(0, 64))
    pos, starts, ends = offset + int(rng.integers(0, 50)), [], []
    while True:
        ln = int(rng.choice([256, 256, 256, int(rng.integers(1, 256))]))
        if pos + ln > n + offset:
            break
        starts.append(pos); ends.append(pos + ln)
        pos += ln + (int(rng.integers(1, 700)) if rng.random() < 0.2 else 0)
    if not starts:
        return
    lens, full = _specagent_fast_slot_setup(starts, ends, offset, sm)
    rl, rf = _reference(starts, ends, offset, sm)
    assert lens == rl and torch.equal(full, rf)
    assert full.data_ptr() != sm.data_ptr() or full.numel() == 0  # copy, like cat


def test_out_of_range_same_error():
    sm = torch.arange(100)
    for starts, ends in (([0, 90], [10, 101]), ([-1], [5])):
        with pytest.raises(ValueError) as a:
            _specagent_fast_slot_setup(starts, ends, 0, sm)
        with pytest.raises(ValueError) as b:
            _reference(starts, ends, 0, sm)
        assert str(a.value) == str(b.value)


def test_reversed_range_falls_back():
    assert _specagent_fast_slot_setup([10], [5], 0, torch.arange(20)) is None
