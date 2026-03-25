#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "attention_kernel.cu"

void run_custom_flash_attn_prefill(
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
    int smem_bytes = (Br + 4 * Bc) * D_padded * sizeof(__nv_bfloat16) + (Br * Bc) * sizeof(float);

    const __nv_bfloat16* q_ptr = reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>());
    const __nv_bfloat16* k_ptr = reinterpret_cast<const __nv_bfloat16*>(k_cache.data_ptr<at::BFloat16>());
    const __nv_bfloat16* v_ptr = reinterpret_cast<const __nv_bfloat16*>(v_cache.data_ptr<at::BFloat16>());
    __nv_bfloat16* o_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
    
    const int* cu_q_ptr = cu_seqlens_q.data_ptr<int>();
    const int* cu_k_ptr = cu_seqlens_k.data_ptr<int>();
    const int* bt_ptr = is_paged? block_table.data_ptr<int>() : nullptr;

    if (head_dim == 128) {
        cudaFuncSetAttribute(flash_attn_prefill_kernel<128, 64, 32>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        flash_attn_prefill_kernel<128, 64, 32><<<grid, block, smem_bytes>>>(
            q_ptr, k_ptr, v_ptr, o_ptr, cu_q_ptr, cu_k_ptr, bt_ptr,
            scale, num_heads, num_kv_heads, block_size, max_blocks_per_seq
        );
    } else if (head_dim == 64) {
        cudaFuncSetAttribute(flash_attn_prefill_kernel<64, 64, 32>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
        flash_attn_prefill_kernel<64, 64, 32><<<grid, block, smem_bytes>>>(
            q_ptr, k_ptr, v_ptr, o_ptr, cu_q_ptr, cu_k_ptr, bt_ptr,
            scale, num_heads, num_kv_heads, block_size, max_blocks_per_seq
        );
    } else {
        TORCH_CHECK(false, "Unsupported head dimension. Only 64 and 128 are supported.");
    }
}

void run_custom_flash_attn_decode(
    torch::Tensor q, 
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor out,
    torch::Tensor cache_seqlens,
    torch::Tensor block_table,
    float scale,
    int n_heads,
    int n_heads_kv,
    int block_size,
    int max_blocks_per_seq
) {   
    TORCH_CHECK(q.dtype() == torch::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(q.dim() == 4, "Input Q must be 4D (batch, seqlen, n_heads, head_dim)");
    TORCH_CHECK(q.size(1) == 1, "Decode kernel only supports seq_len=1");

    int batch_size = q.size(0);
    int seqlen = q.size(1); // 一定为1
    int head_dim = q.size(3);

    int partition_size = 256;
    int max_context_len = max_blocks_per_seq * block_size;
    int num_splits = (max_context_len + partition_size - 1) / partition_size;
    if (num_splits < 1) num_splits = 1;

    // 分配一个shape是(batch_size, n_heads, num_splits, head_dim)的临时空间用于存储reduction所需数据
    auto tmp_options = torch::TensorOptions().dtype(torch::kFloat32).device(q.device());
    torch::Tensor tmp_out = torch::empty({batch_size, n_heads, num_splits, head_dim}, tmp_options);
    torch::Tensor tmp_lse = torch::empty({batch_size, n_heads, num_splits, 2}, tmp_options);

    dim3 grid_stage1(batch_size, n_heads, num_splits);
    dim3 block_stage1(128);
    size_t smem_bytes = head_dim * sizeof(__nv_bfloat16);

    if (head_dim == 64) {
        flash_attn_decode_partial_kernel<64, 256><<<grid_stage1, block_stage1, smem_bytes>>>(
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), // 显式转为 BF16 指针
            reinterpret_cast<const __nv_bfloat16*>(k_cache.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(v_cache.data_ptr()),
            block_table.data_ptr<int>(),
            cache_seqlens.data_ptr<int>(),
            tmp_out.data_ptr<float>(),
            tmp_lse.data_ptr<float>(),
            n_heads, n_heads_kv,
            block_size, max_blocks_per_seq,
            scale
        );
    } else if (head_dim == 128) {
        flash_attn_decode_partial_kernel<128, 256><<<grid_stage1, block_stage1, smem_bytes>>>(
            reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()), // 显式转为 BF16 指针
            reinterpret_cast<const __nv_bfloat16*>(k_cache.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(v_cache.data_ptr()),
            block_table.data_ptr<int>(),
            cache_seqlens.data_ptr<int>(),
            tmp_out.data_ptr<float>(),
            tmp_lse.data_ptr<float>(),
            n_heads, n_heads_kv,
            block_size, max_blocks_per_seq,
            scale
        );
    }

    dim3 grid_stage2(batch_size, n_heads);
    dim3 block_stage2(head_dim);

    if (head_dim == 64) {
        flash_attn_decode_reduce_kernel<64><<<grid_stage2, block_stage2>>>(
            tmp_out.data_ptr<float>(),
            tmp_lse.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            num_splits,
            n_heads
        );
    } else if (head_dim == 128) {
        flash_attn_decode_reduce_kernel<128><<<grid_stage2, block_stage2>>>(
            tmp_out.data_ptr<float>(),
            tmp_lse.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            num_splits,
            n_heads
        );
    }
}

PYBIND11_MODULE(custom_attention_ext, m) {
    m.def("run_custom_flash_attn_prefill", 
            &run_custom_flash_attn_prefill, 
            "Custom Flash Attention Prefill Varlen");
    m.def("run_custom_flash_attn_decode",
            &run_custom_flash_attn_decode,
            "Custom Paged Flash Attention Decode Varlen");
}