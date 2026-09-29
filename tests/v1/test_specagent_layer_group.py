# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from lmcache.v1.memory_management import TensorMemoryObj, MemoryObjMetadata, MemoryFormat
from lmcache.v1.pin_monitor import PinMonitor
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.utils import CacheEngineKey
from lmcache.v1.specagent_layer_group import LayerGroup, chunk_key

class Alloc:
 def __init__(self):self.freed=[]
 def free(self,obj):
  assert obj.is_valid()
  self.freed.append(obj)
  obj.invalidate()

def children(n=4):
 a=Alloc();xs=[]
 for i in range(n):
  t=torch.full((17,2,8),i,dtype=torch.bfloat16)
  xs.append(TensorMemoryObj(t.view(torch.uint8).flatten(),MemoryObjMetadata(t.shape,t.dtype,i,t.numel()*2,1,fmt=MemoryFormat.KV_T2D),a))
 return a,xs

def test_reader_survives_eviction():
 a,xs=children();g=LayerGroup(xs)
 x=g.acquire_layer(2);g.ref_count_down()
 assert len(a.freed)==3 and x.is_valid() and torch.all(x.tensor==2)
 x.ref_count_down();assert len(a.freed)==4

def test_group_pin_owns_all_layers():
 PinMonitor.GetOrCreate(LMCacheEngineConfig.from_defaults())
 a,xs=children();g=LayerGroup(xs);g.pin();g.pin();g.ref_count_down()
 assert not a.freed and not g.can_evict
 g.unpin();assert not a.freed
 g.unpin();assert len(a.freed)==4

def test_cache_and_reader_reference_balance():
 a,xs=children();g=LayerGroup(xs);g.ref_count_up();g.ref_count_down()
 x=g.acquire_layer(0);x.ref_count_down();assert g.can_evict
 g.ref_count_down();assert len(a.freed)==4

def test_keys_preserve_model_tags_and_layer_aliases():
 for cfg in [None,{}, {'lmcache.tag.x':'y'}]:
  k=CacheEngineKey('model',1,0,123,torch.bfloat16,cfg)
  assert all(chunk_key(x)==k for x in k.split_layers(48))

def test_group_size_and_no_implicit_repack():
 a,xs=children();g=LayerGroup(xs)
 assert g.get_size()==4*17*2*8*2 and g.get_num_tokens()==17
 for i,x in enumerate(xs):
  got=g.acquire_layer(i);assert got is x;got.ref_count_down()
 try:g.tensor
 except RuntimeError:pass
 else:raise AssertionError('must forbid implicit bulk access')
 g.ref_count_down()

from collections import OrderedDict
import threading
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend as Backend
class Policy:
 def update_on_put(self,key):pass
 def update_on_force_evict(self,key):pass
 def update_on_hit(self,key,cache):cache.move_to_end(key)

def backend():
 b=Backend.__new__(Backend);b.grouped_layers=True;b.cpu_lock=threading.Lock();b.hot_cache=OrderedDict();b.use_hot=True;b.keys_in_request=[];b.cache_policy=Policy();b.batched_msg_sender=None
 return b

def key(n=123):return CacheEngineKey('m',1,0,n,torch.bfloat16)

def store(b,k):
 a,xs=children();g=LayerGroup(xs);b.batched_submit_put_task([k],[g]);g.ref_count_down();return a,xs,g

def test_complete_owner_is_atomic_for_all_layer_aliases():
 b=backend();k=key();ks=k.split_layers(4)
 assert b.batched_contains(ks)==0
 a,xs,g=store(b,k)
 assert b.batched_contains(ks)==4 and b.contains(k)
 got=b.batched_get_blocking(ks)
 assert got==xs
 for obj in got:obj.ref_count_down()
 b.remove(k);assert not b.contains(k) and len(a.freed)==4

def test_layer_reader_survives_forced_chunk_removal():
 b=backend();k=key();a,xs,g=store(b,k)
 x=b.get_blocking(k.split_layers(4)[2]);assert b.remove(k.split_layers(4)[0])
 assert x.is_valid() and len(a.freed)==3 and torch.all(x.tensor==2)
 x.ref_count_down();assert len(a.freed)==4

def test_pins_count_once_for_base_lookup_but_aliases_balance():
 PinMonitor.GetOrCreate(LMCacheEngineConfig.from_defaults())
 b=backend();k=key();a,xs,g=store(b,k)
 assert b.batched_contains([k],pin=True)==1 and g.metadata.pin_count==1
 b.unpin(k);assert g.metadata.pin_count==0
 assert b.batched_contains(k.split_layers(4),pin=True)==4 and g.metadata.pin_count==4
 for x in k.split_layers(4):b.unpin(x)
 assert g.metadata.pin_count==0;b.touch_cache();b.remove(k)
 assert len(a.freed)==4

def test_missing_chunk_does_not_acquire_reference():
 b=backend();k=key();assert b.get_blocking(k.split_layers(4)[0]) is None
 assert b.batched_get_blocking(k.split_layers(4))==[None]*4

def test_duplicate_publish_releases_only_new_child_ownership():
 b=backend();k=key();a,xs,g=store(b,k);aa,xx,gg=store(b,k)
 assert len(aa.freed)==4 and not a.freed
 x=b.get_blocking(k.split_layers(4)[0]);assert x is xs[0];x.ref_count_down();b.remove(k)

def test_parallel_readers_and_eviction_keep_existing_leases():
 b=backend();k=key();a,xs,g=store(b,k);barrier=threading.Barrier(9);results=[]
 def reader(i):
  x=b.get_blocking(k.split_layers(4)[i%4]);barrier.wait();barrier.wait()
  assert x is not None and x.is_valid() and torch.all(x.tensor==i%4)
  x.ref_count_down();results.append(i)
 ts=[threading.Thread(target=reader,args=(i,)) for i in range(8)]
 for t in ts:t.start()
 barrier.wait();b.remove(k);assert not a.freed;barrier.wait()
 for t in ts:t.join()
 assert len(results)==8 and len(a.freed)==4

from collections import defaultdict
from types import SimpleNamespace as NS,MethodType
from lmcache.v1.storage_backend.storage_manager import StorageManager
import lmcache.v1.cache_engine as module
from lmcache.v1.cache_engine import LMCacheEngine as Engine
def engine():
 b=backend()
 sm=NS(storage_backends={'LocalCPUBackend':b})
 sm.get_active_storage_backends=lambda **kw:iter(sm.storage_backends.items())
 for name in ['batched_contains','batched_unpin','touch_cache']:
  setattr(sm,name,MethodType(getattr(StorageManager,name),sm))
 ks=[key(i) for i in range(3)]
 e=NS(is_healthy=lambda:True, storage_manager=sm,grouped_layers=True,
      token_database=NS(process_tokens=lambda **kw:iter((i*17,(i+1)*17,k) for i,k in enumerate(ks))),
      retrieve_locations=['LocalCPUBackend'],use_layerwise=True,num_layers=4,
      lookup_pins=defaultdict(lambda:defaultdict(list)),
      stats_monitor=NS(on_lookup_request=lambda n:None,on_lookup_finished=lambda *a:None),
      async_loading=False,config=NS(enable_blending=False))
 for n in ['lookup','lookup_unpin']:setattr(e,n,MethodType(getattr(Engine,n),e))
 return e,b,ks

def test_group_lookup_pins_one_owner_per_chunk_and_clears():
 PinMonitor.GetOrCreate(LMCacheEngineConfig.from_defaults())
 e,b,ks=engine();groups=[store(b,k)[2] for k in ks]
 assert e.lookup(tokens=list(range(51)),lookup_id='r',pin=True)==51
 assert [g.metadata.pin_count for g in groups]==[1,1,1]
 e.lookup_unpin('r');assert all(g.metadata.pin_count==0 for g in groups)
 for k in ks:b.remove(k)

def test_missing_group_stops_prefix_and_balances_pins():
 e,b,ks=engine();groups=[store(b,k)[2] for k in ks]
 b.remove(ks[1])
 assert e.lookup(tokens=list(range(51)),lookup_id='r',pin=True)==17
 e.lookup_unpin('r');assert all(g.metadata.pin_count==0 for g in groups)
 for k in ks:b.remove(k)

def test_repeated_lookup_balances_group_pin_ownership():
 e,b,ks=engine();groups=[store(b,k)[2] for k in ks]
 for _ in range(2):assert e.lookup(tokens=list(range(51)),lookup_id='r',pin=True)==51
 assert [g.metadata.pin_count for g in groups]==[2,2,2]
 e.lookup_unpin('r');assert all(g.metadata.pin_count==0 for g in groups)
 for k in ks:b.remove(k)


@pytest.mark.parametrize("fingerprint", [False, True])
def test_store_publishes_all_layers_only_after_copy_and_merge(monkeypatch, fingerprint):
 if fingerprint:monkeypatch.setenv("SPECAGENT_KV_FINGERPRINT", "1")
 else:monkeypatch.delenv("SPECAGENT_KV_FINGERPRINT", raising=False)
 e,b,ks=engine();events=[];allocs=[]
 e.num_layers=4;e.metadata=NS(kv_dtype=torch.bfloat16);e.fmt=module.MemoryFormat.KV_T2D
 e.is_frozen=lambda:False;e._get_req_id=lambda kw:'r';e._log_kvcache_for_check=lambda **kw:None
 e.kv_events_enabled=False;e.config=NS(get_extra_config_value=lambda *args:False)
 e.store_location='LocalCPUBackend'
 e.stats_monitor=NS(on_store_request=lambda n:None,on_store_finished=lambda *a:events.append('finished'))
 e.token_database.chunk_size=17
 e.storage_manager.contains=lambda k,*args:b.contains(k)
 def allocate(*args,**kwargs):
  a,xs=children();allocs.append(a);return xs
 e.storage_manager.batched_allocate=allocate
 def publish(keys,groups,**kwargs):
  assert events[-1]=='fingerprint'
  assert len(keys)==3 and all(len(g.layers)==4 for g in groups)
  b.batched_submit_put_task(keys,groups)
  for g in groups:g.ref_count_down()
  events.append('publish')
 e.storage_manager.batched_put=publish
 def copy(objects,starts,ends,**kwargs):
  assert not b.hot_cache
  for layer in objects:
   for obj in layer:assert obj.is_valid()
  events.append('copy_done')
  for i in range(5):yield
 e.gpu_connector=NS(get_shape=lambda n:torch.Size([n,2,8]),batched_from_gpu=copy)
 monkeypatch.setattr(module,'assert_layerwise_gpu_connector',lambda c:None)
 monkeypatch.setattr(module,'_specagent_record_actual_kv_fingerprint',lambda *a:events.append('fingerprint'))
 def proof(*args,**kwargs):
  assert all(b.contains(k) for k in ks)
  events.append('proof')
 monkeypatch.setattr(module,'_specagent_record_actual_cpu_store',proof)
 list(Engine.store_layer(e,list(range(51)),specagent_defer_layerwise_store_commit=True))
 assert events.index('copy_done')<events.index('publish')<events.index('proof')
 assert events.count('fingerprint') == 12
 for k in ks:
  xs=b.batched_get_blocking(k.split_layers(4));assert all(x.is_valid() for x in xs)
  for x in xs:x.ref_count_down()
  b.remove(k)
 assert all(len(a.freed)==4 for a in allocs)


def test_group_batch_acquisition_survives_eviction_and_balances_leases():
 b=backend();ks=[key(i) for i in range(3)];stored=[store(b,k) for k in ks]
 got=b.batched_get_layer_groups(ks,4)
 assert got is not None and len(got)==4
 for layer in range(4):
  assert got[layer]==[entry[1][layer] for entry in stored]
 for k in ks:b.remove(k)
 assert all(not a.freed for a,_,_ in stored)
 for layer in got:
  for obj in layer:assert obj.is_valid();obj.ref_count_down()
 assert all(len(a.freed)==4 for a,_,_ in stored)


def test_group_batch_missing_or_wrong_layer_count_rolls_back():
 b=backend();k=key();a,xs,g=store(b,k)
 assert b.batched_get_layer_groups([k,key(999)],4) is None
 assert all(x.metadata.ref_count==1 for x in xs)
 assert b.batched_get_layer_groups([k],3) is None
 assert all(x.metadata.ref_count==1 for x in xs)
 b.remove(k);assert len(a.freed)==4


def test_prepared_group_segments_transfer_and_release_exact_leases():
 e,b,ks=engine();stored=[store(b,k) for k in ks]
 e.bulk_validate_sparse_layerwise_plan=MethodType(Engine.bulk_validate_sparse_layerwise_plan,e)
 plans=[[(0,17,ks[0]),(17,34,ks[1])],[(34,51,ks[2])]]
 batch=Engine.prepare_sparse_layerwise_plans(e,plans)
 assert batch is not None
 segment=batch.claim(0)
 assert segment.starts==[0,17] and segment.ends==[17,34]
 assert segment.keys_by_chunk==[k.split_layers(4) for k in ks[:2]]
 batch.release_unclaimed()
 for k in ks:b.remove(k)
 assert len(stored[2][0].freed)==4
 assert not stored[0][0].freed and not stored[1][0].freed
 for layer in segment.memory_objs_by_layer:
  for obj in layer:assert obj.is_valid();obj.ref_count_down()
 assert all(len(a.freed)==4 for a,_,_ in stored)


def test_prepare_group_evicted_after_validation_does_not_leak(monkeypatch):
 e,b,ks=engine();stored=[store(b,k) for k in ks]
 e.bulk_validate_sparse_layerwise_plan=MethodType(Engine.bulk_validate_sparse_layerwise_plan,e)
 original=b.batched_get_layer_groups
 def evict_then_acquire(keys,n):
  b.remove(ks[1]);return original(keys,n)
 monkeypatch.setattr(b,'batched_get_layer_groups',evict_then_acquire)
 with pytest.raises(RuntimeError,match='disappeared'):
  Engine.prepare_sparse_layerwise_plans(e,[[(i*17,(i+1)*17,k) for i,k in enumerate(ks)]])
 for k in ks:b.remove(k)
 assert all(len(a.freed)==4 for a,_,_ in stored)
