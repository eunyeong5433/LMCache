// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <torch/all.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>
#include <cstdint>
#include <vector>

enum class TransferDirection : int {
  H2D = 0,
  D2H = 1,
};

/*
Symbol Reference:
NL: number of layers
NB: number of blocks/pages
BS: block/page size
NBBS: block/page buffer size = NB * BS
NH: number of heads
HS: head size
TWO: 2
ONE: 1

_ means a dimension within the same tensor
_X_ means a dimension across a list

A_X_B_X_C_D_E means:
kv_cache: List[List[torch.Tensor]]
len(kv_cache) = A
len(kv_cache[0]) = B
kv_cache[0][0].shape = (C, D, E)

The logic for identifying the format currently lives in
`lmcache/v1/gpu_connector/utils.py`
*/
enum class GPUKVFormat : int {
  NB_NL_TWO_BS_NH_HS = 0,
  /*
  used by:
  - vLLM CROSS_LAYER mode
  */

  NL_X_TWO_NB_BS_NH_HS = 1,
  /*
  used by:
  - vLLM non-MLA flash attention
  */

  NL_X_NB_TWO_BS_NH_HS = 2,
  /*
  used by:
  - vLLM non-MLA flash infer
  */

  NL_X_NB_BS_HS = 3,
  /*
  used by:
  - vLLM MLA
  */

  TWO_X_NL_X_NBBS_NH_HS = 4,
  /*
  used by:
  - SGLang MHA (flash attention and flash infer)
  */

  NL_X_NBBS_ONE_HS = 5,
  /*
  used by:
  - SGLang MLA
  */

  NL_X_TWO_NB_NH_BS_HS = 6,
  /*
  used by:
  - vLLM non-MLA flash attention (HND layout)
  physical shape per layer: [2, num_blocks, num_heads, block_size, head_size]
  */

  NL_X_NB_TWO_NH_BS_HS = 7,
  /*
  used by:
  - vLLM non-MLA flash infer (HND layout)
  physical shape per layer: [num_blocks, 2, num_heads, block_size, head_size]
  */
};

void multi_layer_kv_transfer(
    torch::Tensor& key_value, const torch::Tensor& key_value_ptrs,
    const torch::Tensor& slot_mapping, const torch::Device& paged_memory_device,
    const int page_buffer_size, const TransferDirection direction,
    const GPUKVFormat gpu_kv_format, const int block_size = 0,
    const int head_size = 0, const int skip_prefix_n_tokens = 0);

// collapses to multi_layer_kv_transfer for MLA
void multi_layer_kv_transfer_unilateral(
    torch::Tensor& key_value, const torch::Tensor& key_value_ptrs,
    const torch::Tensor& slot_mapping, const torch::Device& paged_memory_device,
    const int page_buffer_size, const TransferDirection direction,
    const GPUKVFormat gpu_kv_format);

void single_layer_kv_transfer(torch::Tensor& lmc_key_value_cache,
                              torch::Tensor& vllm_key_value_cache,
                              torch::Tensor& slot_mapping,
                              const TransferDirection direction,
                              const GPUKVFormat gpu_kv_format,
                              const bool token_major = false);

void single_layer_kv_transfer_sgl(torch::Tensor& lmc_key_value_cache,
                                  torch::Tensor& sgl_key_cache,
                                  torch::Tensor& sgl_value_cache,
                                  torch::Tensor& slot_mapping,
                                  const TransferDirection direction,
                                  const bool token_major = false);

void lmcache_memcpy_async(uintptr_t dest, uintptr_t src, size_t nbytes,
                          TransferDirection direction,
                          size_t host_buffer_offset,
                          size_t host_buffer_alignments);

void lmcache_memcpy_async_batch(const std::vector<uintptr_t>& dests,
                                const std::vector<uintptr_t>& srcs,
                                const std::vector<size_t>& nbytes,
                                TransferDirection direction,
                                const std::vector<size_t>& host_buffer_offsets,
                                const std::vector<size_t>& host_buffer_alignments);

// CUDA 12.8+ scatter/gather batch-copy API.  Unlike
// lmcache_memcpy_async_batch, this is one runtime submission for the whole
// independent pointer list instead of one cudaMemcpyAsync call per entry.
void lmcache_memcpy_batch_async(const std::vector<uintptr_t>& dests,
                                const std::vector<uintptr_t>& srcs,
                                const std::vector<size_t>& nbytes,
                                TransferDirection direction);

// Submit all disjoint scratch scatters for one model layer from one Python
// call.  The caller may batch H2D first and insert the scratch-slot WAR event
// before invoking this function.
void specagent_multi_scatter(
    const std::vector<torch::Tensor>& transfer_tensors,
    const std::vector<torch::Tensor>& target_kv_caches,
    const std::vector<torch::Tensor>& slot_mappings,
    const GPUKVFormat gpu_kv_format);

// deprecated / unused except in unit tests
void load_and_reshape_flash(torch::Tensor& key_value, torch::Tensor& key_cache,
                            torch::Tensor& value_cache,
                            torch::Tensor& slot_mapping, const int layer_idx);

// deprecated / unused except in unit tests
void reshape_and_cache_back_flash(torch::Tensor& key_value,
                                  torch::Tensor& key_cache,
                                  torch::Tensor& value_cache,
                                  torch::Tensor& slot_mapping,
                                  const int layer_idx);
