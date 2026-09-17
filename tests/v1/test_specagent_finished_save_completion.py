# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace as NS

import pytest
import torch

from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl


@pytest.mark.parametrize("preempt", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_finished_save_drains_before_success(preempt, fail):
    events = []

    def store_layer(tokens, **kwargs):
        assert kwargs['specagent_defer_layerwise_store_commit'] is True
        assert kwargs['specagent_save_current_layer_only'] is False
        assert kwargs['sync'] is True
        for _ in range(3):
            yield
        if fail:
            raise RuntimeError('copy failed')
        events.append('all_copies_completed_and_published')
        yield

    adapter = NS(kv_role='kv_both', device='cpu', _lmcache_chunk_size=16,
                 _block_size=16, kv_caches={'0': torch.zeros(1)},
                 lmcache_engine=NS(store_layer=store_layer))
    req = NS(is_finished_save=not preempt, is_preempt_save=preempt,
             req_id='test', token_ids=list(range(19)), slot_mapping=torch.arange(19),
             save_spec=NS(can_save=True, skip_leading_tokens=0))
    call = LMCacheConnectorV1Impl._specagent_save_finished_meta_without_forward
    if fail:
        with pytest.raises(RuntimeError, match='copy failed'):
            call(adapter, req)
        assert not events
    else:
        assert call(adapter, req) is True
        assert events == ['all_copies_completed_and_published']


@pytest.mark.parametrize('layout', ['NHD', 'HND'])
def test_deferred_store_actual_gpu_data(layout):
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemLayerwiseGPUConnector
    connector = VLLMPagedMemLayerwiseGPUConnector(
        16, 3, use_gpu=True, dtype=torch.bfloat16, device=torch.device('cuda'),
        chunk_size=16, layout_hints={'kv_layout': layout})
    caches = []
    for _ in range(3):
        physical = torch.randn((2, 4, 16, 2, 8) if layout == 'NHD' else
                               (2, 4, 2, 16, 8), device='cuda', dtype=torch.bfloat16)
        caches.append(physical if layout == 'NHD' else physical.permute(0, 1, 3, 2, 4))
    ranges = [(0, 16), (16, 19)]
    objects = [[NS(tensor=torch.empty(end-start, 2, 16, dtype=torch.bfloat16,
                                     pin_memory=True)) for start, end in ranges]
               for _ in range(3)]
    list(connector.batched_from_gpu(objects, [0, 16], [16, 19],
         kvcaches=caches, slot_mapping=torch.arange(19, device='cuda'), sync=True,
         specagent_save_current_layer_only=False,
         specagent_defer_store_tmp_buffer_free=False,
         specagent_defer_layerwise_store_commit=True))
    for cache, chunks in zip(caches, objects):
        expected = cache.cpu().reshape(2, 64, 16).permute(1, 0, 2)
        for (start, end), obj in zip(ranges, chunks):
            torch.testing.assert_close(obj.tensor, expected[start:end], rtol=0, atol=0)
