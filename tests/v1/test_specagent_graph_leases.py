# SPDX-License-Identifier: Apache-2.0
"""Graph fetch owners survive until completion, without an ordinary fetch."""

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemLayerwiseGPUConnector


class Completion:
    def __init__(self) -> None:
        self.ready = False

    def query(self) -> bool:
        return self.ready


class Lease:
    def __init__(self) -> None:
        self.releases = 0

    def ref_count_down(self) -> None:
        self.releases += 1


@pytest.mark.no_shared_allocator
def test_consecutive_graph_steps_release_completed_owners(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise public retire/reap APIs; no real CUDA work is needed."""
    monkeypatch.delenv("SPECAGENT_LOW_PRIORITY_LOAD_STREAM", raising=False)
    monkeypatch.setattr(torch.cuda, "Stream", lambda: object())
    connector = VLLMPagedMemLayerwiseGPUConnector(
        1024, 2, chunk_size=16, dtype=torch.bfloat16, device="cpu"
    )
    for _ in range(2):
        complete = Completion()
        owner_lease = Lease()
        source_leases = [Lease(), Lease()]
        owner = {"memory_objs": [owner_lease]}
        plans = [
            [
                {
                    "layer_events": [None, None],
                    "source_lease_objs": [source_leases[index]],
                    "source_owner": owner,
                }
            ]
            for index in range(2)
        ]
        connector.specagent_retire_graph_plans(complete, plans)
        connector.specagent_reap_completed_plans()
        assert owner_lease.releases == 0
        assert all(lease.releases == 0 for lease in source_leases)
        assert all(plans[i][0]["layer_events"][i] is complete for i in range(2))
        complete.ready = True
        connector.specagent_reap_completed_plans()
        connector.specagent_reap_completed_plans()
        assert owner_lease.releases == 1
        assert all(lease.releases == 1 for lease in source_leases)
