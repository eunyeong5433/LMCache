# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

# Standard
import collections
import threading

# Third Party
import pytest
import torch

# First Party
from lmcache.utils import CacheEngineKey
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend


class Memory:
    def __init__(self):
        self.pins = 0

    def pin(self):
        self.pins += 1
        return True


@pytest.mark.parametrize("pin", [False, True])
@pytest.mark.parametrize("stop", [0, 1, 32, 63, 64])
@pytest.mark.parametrize("duplicate", [False, True])
def test_prefix_and_pin_contract(pin, stop, duplicate):
    keys = CacheEngineKey("test", 1, 0, 123, torch.bfloat16).split_layers(64)
    sequence = keys[:16] * 2 + keys[16:] if duplicate else keys
    objects = [Memory() for _ in keys]
    backend = SimpleNamespace(
        cpu_lock=threading.Lock(),
        hot_cache=collections.OrderedDict(
            zip(keys[:stop], objects[:stop], strict=False)
        ),
        keys_in_request=[],
    )
    expected = []
    for key in sequence:
        if key not in keys[:stop]:
            break
        expected.append(key)
    assert LocalCPUBackend.batched_contains(backend, sequence, pin) == len(expected)
    assert backend.keys_in_request == (expected if pin else [])
    assert [o.pins for o in objects] == [
        expected.count(key) if pin else 0 for key in keys
    ]
    assert list(backend.hot_cache) == keys[:stop]


def test_empty_batch():
    backend = SimpleNamespace(
        cpu_lock=threading.Lock(), hot_cache={}, keys_in_request=[]
    )
    assert LocalCPUBackend.batched_contains(backend, [], True) == 0
    assert backend.keys_in_request == []
