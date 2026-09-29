# SPDX-License-Identifier: Apache-2.0
"""Native turn-end/preempt saves complete once before their public ACK."""
# Standard
from types import SimpleNamespace as NS
from typing import Any
# Third Party
import pytest
import torch
# First Party
from lmcache.integration.vllm import vllm_v1_adapter as module
from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorMetadata, LMCacheConnectorV1Impl


def make_adapter(kind: str, length: int, events: list[str], fail: bool = False) -> LMCacheConnectorV1Impl:
    req = NS(req_id='r', token_ids=list(range(length)), slot_mapping=torch.arange(length),
             save_spec=NS(can_save=True, skip_leading_tokens=16),
             is_finished_save=kind=='finished', is_preempt_save=kind=='preempt',
             specagent_preempt_save_id=7, specagent_tail_merge_prefix_tokens=0,
             specagent_observation_tail_save=False, is_last_prefill=True,
             disagg_spec=None, request_configs=None)
    meta = LMCacheConnectorMetadata(requests=[req])
    adapter = object.__new__(LMCacheConnectorV1Impl)
    def store(tokens: list[int], **kwargs: Any) -> None:
        assert not getattr(adapter, '_specagent_finished_save_completed')
        assert not getattr(adapter, '_specagent_preempt_save_completed')
        assert kwargs['mask'][:16].tolist() == [False]*16
        assert kwargs['mask'][16:].all()
        assert len(tokens) == (length//16*16 if kind=='ordinary' else length)
        if fail:raise RuntimeError('copy failed')
        events.append('completed_and_published')
    engine = NS(native_bulk_spec=True, store=store, lookup_unpin=lambda req_id:None,
                lookup=lambda **kwargs:length)
    fields=dict(_parent=NS(_get_connector_metadata=lambda:meta),kv_role='kv_both',
                use_layerwise=False,device='cpu',_lmcache_chunk_size=16,_block_size=16,
                kv_caches={'0':torch.zeros(1)},_manager=NS(lmcache_engine=engine),enable_blending=False,
                _specagent_finished_save_completed=set(),_specagent_preempt_save_completed={})
    for k,v in fields.items():setattr(adapter,k,v)
    return adapter


@pytest.mark.parametrize('kind', ['finished', 'preempt', 'ordinary'])
@pytest.mark.parametrize('length', [32, 37])
@pytest.mark.parametrize('no_forward', [False, True])
def test_store_once_and_ack_after_publication(monkeypatch: Any, kind: str, length: int, no_forward: bool) -> None:
    if no_forward and kind == 'ordinary':pytest.skip('ordinary save requires a forward')
    events: list[str] = []
    adapter=make_adapter(kind,length,events)
    monkeypatch.setattr(module,'get_pp_group',lambda:NS(is_last_rank=True))
    if not no_forward:
        adapter.wait_for_save()
        assert events == (['completed_and_published'] if kind=='ordinary' else [])
    sending,receiving=adapter.get_finished({'r'} if kind=='finished' else set())
    assert events == ['completed_and_published']
    assert receiving is None
    assert sending == ({'r'} if kind=='finished' else None)
    worker_meta=adapter.build_connector_worker_meta()
    if kind=='preempt':assert worker_meta.completed_preempt_saves=={('r',7):(length,1)}
    else:assert worker_meta is None


def test_failed_native_copy_is_not_acknowledged() -> None:
    events: list[str] = []
    adapter=make_adapter('finished',19,events,fail=True)
    adapter.wait_for_save()
    with pytest.raises(RuntimeError,match='copy failed'):adapter.get_finished({'r'})
    assert not events
    assert not getattr(adapter,'_specagent_finished_save_completed')
