# SPDX-License-Identifier: Apache-2.0
"""A resumed SpecAgent fetch request keeps its whole computed prefix.

``specagent_tracker_update_counts`` picks the counts that
``build_connector_meta`` passes to ``RequestTracker.update``. For a SpecAgent
fetch the tracker keeps its own save progress, while its token list covers the
prefix the scheduler treats as computed (CPU-fetched holes included). The
numbers are those of the K2 m350 crash: a 217,705-token request whose tracker
had saved 31,640 tokens came out 34,921 tokens long.
"""

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import (
    RequestTracker,
    specagent_tracker_update_counts,
)

COMPUTED = 214_424
NEW = 3_281
SAVED = 31_640


def _resume(is_fetch: bool) -> RequestTracker:
    all_tokens = list(range(COMPUTED + NEW))
    tracker = RequestTracker(
        req_id="r",
        prompt_len=COMPUTED + NEW,
        token_ids=all_tokens[:SAVED],
        allocated_block_ids=[],
        num_saved_tokens=SAVED,
    )
    lmcache_cached, vllm_cached = specagent_tracker_update_counts(
        is_fetch, tracker.num_saved_tokens, 182_784, 0, COMPUTED
    )
    tracker.update(
        all_tokens[COMPUTED : COMPUTED + NEW],
        [],
        preempted=True,
        lmcache_cached_tokens=lmcache_cached,
        vllm_cached_tokens=vllm_cached,
        all_token_ids=all_tokens,
    )
    return tracker


def test_fetch_resume_keeps_computed_prefix_and_save_progress():
    tracker = _resume(is_fetch=True)
    assert len(tracker.token_ids) == COMPUTED + NEW
    assert tracker.token_ids[-NEW:] == list(range(COMPUTED, COMPUTED + NEW))
    assert tracker.num_saved_tokens == SAVED


def test_non_fetch_counts_are_unchanged():
    assert specagent_tracker_update_counts(False, SAVED, 182_784, 7, COMPUTED) == (
        182_784,
        7,
    )


def test_fetch_counts():
    assert specagent_tracker_update_counts(True, SAVED, 182_784, 0, COMPUTED) == (
        SAVED,
        COMPUTED,
    )
    # A larger vLLM-cached count is kept.
    assert specagent_tracker_update_counts(True, 0, 0, COMPUTED + 5, COMPUTED) == (
        0,
        COMPUTED + 5,
    )
