# SPDX-License-Identifier: Apache-2.0
"""Exercise production mixed dispatch with actual CUDA copies per layer."""
import ast
import logging
import os
import threading
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
import torch
p=Path('/home/eunyeong/gh200-migration/lmcache-gh200-final/lmcache/integration/vllm/vllm_v1_adapter.py')
t=ast.parse(p.read_text());names={'_specagent_advance_retrievers_until','_specagent_events_ready','_specagent_has_layerwise_fetch'}
methods=[n for c in t.body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef) and n.name in names]
module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),ast.ClassDef(name='Adapter',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[])
scope=dict(torch=torch,os=os,logger=logging.getLogger('gpu-test'),record_function_or_nullcontext=lambda *a:nullcontext());exec(compile(ast.fix_missing_locations(module),str(p),'exec'),scope)
for normal_first in [True,False]:
 a=scope['Adapter']();a.num_layers=64;a._specagent_prefetch_scan_layer=0;a._specagent_normal_next_layer={};a._specagent_prefetch_cond=threading.Condition();a.lmcache_engine=NS(gpu_connector=NS())
 source=torch.arange(64*32,device='cuda').reshape(64,32);normal=torch.full_like(source,-1);spec=torch.full_like(source,-1);events=[None]*64;released=[]
 def normal_load():
  for i in range(64):
   yield None
   normal[i].copy_(source[i])
  yield None
  torch.cuda.synchronize();released.append(True)
  yield 2048
 def spec_load():
  for i in range(64):
   spec[i].copy_(source[i]);events[i]=object();yield None
 n=normal_load();next(n);next(n)
 pair=[(n,None),(spec_load(),dict(is_specagent=True,expected_tokens=0,layer_events=events))]
 if not normal_first:pair.reverse()
 a.layerwise_retrievers=[x[0] for x in pair];a._specagent_layerwise_meta=[x[1] for x in pair]
 for i in range(64):
  a._specagent_advance_retrievers_until(i)
  assert torch.equal(normal[i],source[i]),('normal layer not ready',i)
  assert torch.equal(spec[i],source[i]),('spec layer not ready',i)
 assert released==[True]
 a._specagent_advance_retrievers_until(63);assert released==[True]
 print('PASS CUDA mixed NORMAL/SPEC every layer correct, finalized once; normal_first=',normal_first,flush=True)
