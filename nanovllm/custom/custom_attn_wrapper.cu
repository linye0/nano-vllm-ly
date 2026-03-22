#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "attention_kernel.cu"

void run_custom_flash_attn_varlen(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor out,
    torch::Tensor cu_seqlens_q,
    torch::Tensor cu_seqlens_k,
    torch::Tensor block_table,
    bool is_paged,
    float scale,
    int max_seqlen_q,
    int max_seqlen_k,
    int num_heads,
    int num_kv_heads,
    int block_size,
    int max_blocks_per_seq
) {
    TORCH_CHECK(q.dtype() == torch::kBFloat16, "q must be bfloat16");
    
    int batch_size = cu_seqlens_q.size(0) - 1;
    int head_dim = q.size(2);

    int Br = 64;
    int Bc = 32;

    int num_q_chunks = (max_seqlen_q + Br - 1) / Br;
    
    dim3 grid(num_q_chunks, num_heads, batch_size);
    dim3 block(128);

    int D_padded = head_dim + 8;
    // 🌟 核心更新：加入 P 中转站所需的大小
    int smem_bytes = (Br + 4 * Bc) * D_padded * sizeof(__nv_bfloat16) + (Br * Bc) * sizeof(float);

    const __nv_bfloat16* q_ptr = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>());
    const __nv_bfloat16* k_ptr = reinterpret_cast<const __nv_bfloat16*>(k_cache.data_ptr<at::BFloat16>());
    const __nv_bfloat16* v_ptr = reinterpret_cast<const __nv_bfloat16*>(v_cache.data_ptr<at::BFloat16>());
    __nv_bfloat16* o_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
    
    const int* cu_q_ptr = cu_seqlens_q.data_ptr<int>();
    const int* cu_k_ptr = cu_seqlens_k.data_ptr<int>();
    const int* bt_ptr = is_paged? block_table.data_ptr<int>() : nullptr;

    if (head_dim == 128) {
        cudaFuncSetAttribute(flash_attn_varlen_wmma_kernel<128, 64, 32>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        flash_attn_varlen_wmma_kernel<128, 64, 32><<<grid, block, smem_bytes>>>(
            q_ptr, k_ptr, v_ptr, o_ptr, cu_q_ptr, cu_k_ptr, bt_ptr,
            scale, num_heads, num_kv_heads, block_size, max_blocks_per_seq
        );
    } else if (head_dim == 64) {
        cudaFuncSetAttribute(flash_attn_varlen_wmma_kernel<64, 64, 32>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        flash_attn_varlen_wmma_kernel<64, 64, 32><<<grid, block, smem_bytes>>>(
            q_ptr, k_ptr, v_ptr, o_ptr, cu_q_ptr, cu_k_ptr, bt_ptr,
            scale, num_heads, num_kv_heads, block_size, max_blocks_per_seq
        );
    } else {
        TORCH_CHECK(false, "Unsupported head dimension. Only 64 and 128 are supported.");
    }
}

PYBIND11_MODULE(custom_attention_ext, m) {
    m.def("run_custom_flash_attn_varlen", &run_custom_flash_attn_varlen, "Custom Paged Flash Attention Varlen");
}