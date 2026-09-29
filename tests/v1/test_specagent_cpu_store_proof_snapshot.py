# SPDX-License-Identifier: Apache-2.0
"""CPU proof publication must retain exact prefix keys with shared snapshots."""
# Standard
import ast
import builtins
import logging
import os
from pathlib import Path
from typing import Optional, Union

# Third Party
import pytest
import torch


def load_proof_recorder():
    # Isolate the CPU ledger helper from engine/allocator initialization.
    source = Path(__file__).resolve().parents[2] / "lmcache/v1/cache_engine.py"
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "_specagent_record_actual_cpu_store")
    scope = dict(builtins=builtins, os=os, torch=torch, Union=Union,
                 Optional=Optional, logger=logging.getLogger(__name__))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
    return scope[node.name]


@pytest.mark.parametrize("tensor_input", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("length", [15, 256, 773])
def test_snapshot_publishes_exact_completed_prefixes(monkeypatch, tensor_input, legacy, length):
    monkeypatch.delenv("SPECAGENT_DISABLE_CPU_STORE_PROOF", raising=False)
    if legacy:
        monkeypatch.setenv("SPECAGENT_LEGACY_PRESTORE_PROOF", "1")
    else:
        monkeypatch.delenv("SPECAGENT_LEGACY_PRESTORE_PROOF", raising=False)
    monkeypatch.setattr(builtins, "_specagent_cpu_saved_chunks", set(), raising=False)
    monkeypatch.setattr(builtins, "_specagent_cpu_saved_blocks", set(), raising=False)
    tokens = [i * 17 - 100 for i in range(length)]
    original = list(tokens)
    snapshot = tuple(tokens)
    inputs = torch.tensor(tokens) if tensor_input else tokens
    hashes = {}
    recorder = load_proof_recorder()
    ends = list(range(256, length, 256)) + [length]
    previous = 0
    for end in ends:
        recorder(inputs, previous, end, 16, prefix_snapshot=snapshot, prefix_hashes=hashes)
        previous = end
    assert builtins._specagent_cpu_saved_chunks == {
        (end, hash(tuple(tokens[:end]))) for end in ends
    }
    assert hashes == {end: hash(tuple(tokens[:end])) for end in ends}
    assert tokens == original
    expected_blocks = {(end // 16 - 1, hash(tuple(tokens[:end])))
                       for end in range(16, length + 1, 16)} if legacy else set()
    assert builtins._specagent_cpu_saved_blocks == expected_blocks


def test_disabled_proof_does_not_publish_snapshot(monkeypatch):
    monkeypatch.setenv("SPECAGENT_DISABLE_CPU_STORE_PROOF", "1")
    ledger = set()
    monkeypatch.setattr(builtins, "_specagent_cpu_saved_chunks", ledger, raising=False)
    hashes = {}
    load_proof_recorder()([1, 2], 0, 2, 16, prefix_snapshot=(1, 2), prefix_hashes=hashes)
    assert not ledger and not hashes
