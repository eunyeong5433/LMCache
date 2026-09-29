# SPDX-License-Identifier: Apache-2.0
"""Actual CPU bytes, sparse slots and long mappings for compatible D2H."""
from types import SimpleNamespace as NS
import pytest
import torch
from lmcache.v1.specagent_direct_store import store_paged_layers

@pytest.mark.parametrize('layout', ['NHD', 'HND'])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('base', [0, 70000])
def test_direct_store_exact(layout, dtype, base):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    count,block,heads,dim,nlayers=37,16,2,16,3
    slots=torch.full((base+count,),-1,dtype=torch.int64,device='cuda')
    slots[base:]=torch.arange(count,device='cuda')+16
    slots[base+9]=-1
    caches=[]
    for layer in range(nlayers):
     shape=(2,5,block,heads,dim) if layout=='NHD' else (2,5,heads,block,dim)
     p=torch.randn(shape,dtype=dtype,device='cuda')
     caches.append(p if layout=='NHD' else p.permute(0,1,3,2,4))
    starts=[base,base+16,base+32];ends=[base+16,base+32,base+37]
    outputs=[[NS(tensor=torch.full((e-s,2,heads*dim),-7,dtype=dtype,pin_memory=True)) for s,e in zip(starts,ends)] for _ in caches]
    store_paged_layers(outputs,starts,ends,caches,slots,torch.cuda.Stream())
    for cache,objects in zip(caches,outputs):
     for obj,s,e in zip(objects,starts,ends):
      for i in range(s,e):
       slot=int(slots[i]);expected=cache[:,slot//block,slot%block].cpu().reshape(2,-1) if slot>=0 else torch.full((2,heads*dim),-7,dtype=dtype)
       torch.testing.assert_close(obj.tensor[i-s],expected,rtol=0,atol=0)


def test_wait_for_save_completes_all_layers_before_publication_and_unpin(monkeypatch):
    from lmcache.integration.vllm.vllm_v1_adapter import (
        LMCacheConnectorV1Impl, LMCacheConnectorMetadata,
    )
    monkeypatch.setenv('SPECAGENT_DIRECT_BULK_SAVE', '1')
    events = []
    def storer():
        for layer in range(3):
            events.append(('write', layer))
            yield
        events.append('publish')
        yield
    request = NS(req_id='r', is_finished_save=True)
    meta = LMCacheConnectorMetadata(requests=[request])
    state = NS(_parent=NS(_get_connector_metadata=lambda: meta),
               kv_role='kv_both', use_layerwise=True,
               _specagent_join_save_futures=lambda: None,
               _layerwise_save_storers={'r': storer()},
               _specagent_finished_save_completed=set(),
               lmcache_engine=NS(lookup_unpin=lambda req: events.append('unpin')))
    LMCacheConnectorV1Impl.wait_for_save(state)
    assert events == [('write',0),('write',1),('write',2),'publish','unpin']
    assert state._specagent_finished_save_completed == {'r'}
    assert not state._layerwise_save_storers


def test_early_layer_hook_never_reads_later_kv(monkeypatch):
    from lmcache.integration.vllm.vllm_v1_adapter import (
        LMCacheConnectorV1Impl, LMCacheConnectorMetadata,
    )
    monkeypatch.setenv('SPECAGENT_DIRECT_BULK_SAVE', '1')
    monkeypatch.delenv('SPECAGENT_KV_FINGERPRINT_PRESTORE', raising=False)
    meta = LMCacheConnectorMetadata(requests=[NS()])
    state = NS(lmcache_engine=NS(), use_layerwise=True, kv_role='kv_both',
               _parent=NS(_connector_metadata=meta, _get_connector_metadata=lambda:meta),
               kv_caches={'first':None}, num_layers=3)
    # The deliberately incomplete request would fail if traversed too early.
    LMCacheConnectorV1Impl.save_kv_layer(state, 'model.layers.0.self_attn', None, None)

