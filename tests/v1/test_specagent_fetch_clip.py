# SPDX-License-Identifier: Apache-2.0
"""SpecAgent fetch ranges the adapter cannot cover are counted, not dropped silently.

``specagent_fetch_lost_tokens`` measures the requested fetch tokens at or past
the adapter's cover limit (token list, slot mapping, LMCache prefix, cascade
prefix). With the short tracker of the K2 m350 crash (34,921 tokens for a
217,705-token request) the holes past 34,921 were dropped without a log.
"""

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import specagent_fetch_lost_tokens


def test_ranges_inside_the_limit_lose_nothing():
    assert specagent_fetch_lost_tokens([(0, 256), (512, 1024)], 1024) == 0
    assert specagent_fetch_lost_tokens([], 0) == 0


def test_short_tracker_drops_the_holes_past_it():
    ranges = [(16, 4_096), (30_000, 40_000), (100_000, 214_416)]
    assert specagent_fetch_lost_tokens(ranges, 34_921) == (
        (40_000 - 34_921) + (214_416 - 100_000)
    )


def test_range_ending_at_the_limit_is_covered():
    assert specagent_fetch_lost_tokens([(0, 4_096)], 4_096) == 0
    assert specagent_fetch_lost_tokens([(0, 4_112)], 4_096) == 16
