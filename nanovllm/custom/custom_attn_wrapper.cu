#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// 1. 直接把你写好的 kernel 文件包含进来
// 这样 nvcc 在编译时就能完全看到 __global__ 函数和模板，不会报 unresolved 错误
#include "attention_kernel.cu"

// 2. C++ 包装函数
void run_custom_flash_attn_varlen(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor out,
    torch::Tensor cu_seqlens_q,
    torch::Tensor cu_seqlens_k,
    torch::Tensor block_table,
    float scale,
    int max_seqlen_q,
    int max_seqlen_k,
    int num_heads,
    int num_kv_heads,
    int block_size,
    int max_blocks_per_seq
) {
    // 确保输入是 FP16
    TORCH_CHECK(q.dtype() == torch::kFloat16, "q must be float16");
    
    int batch_size = cu_seqlens_q.size(0) - 1;
    int head_dim = q.size(2);

    // Br = 64, Bc = 32
    int Br = 64;
    int Bc = 32;

    // 计算 Grid 和 Block 维度
    int num_q_chunks = (max_seqlen_q + Br - 1) / Br;
    
    // Grid: [q_chunk_idx, head_id, batch_id]
    dim3 grid(num_q_chunks, num_heads, batch_size);
    
    // Block: 4个 Warp (128 threads) 负责 64 行
    dim3 block(128);

    int D_padded = head_dim + 8;
    int smem_bytes = (Br + 4 * Bc) * D_padded * sizeof(half);

    // 提取原始指针
    const half* q_ptr = reinterpret_cast<const half*>(q.data_ptr<at::Half>());
    const half* k_ptr = reinterpret_cast<const half*>(k_cache.data_ptr<at::Half>());
    const half* v_ptr = reinterpret_cast<const half*>(v_cache.data_ptr<at::Half>());
    half* o_ptr = reinterpret_cast<half*>(out.data_ptr<at::Half>());
    
    const int* cu_q_ptr = cu_seqlens_q.data_ptr<int>();
    const int* cu_k_ptr = cu_seqlens_k.data_ptr<int>();
    const int* bt_ptr = block_table.data_ptr<int>();

    // 根据 Head Dim 模板派发并启动 Kernel
    if (head_dim == 128) {
        // 先提额，再刷卡
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

// 3. 绑定到 Python
PYBIND11_MODULE(custom_attention_ext, m) {
    m.def("run_custom_flash_attn_varlen", &run_custom_flash_attn_varlen, "Custom Paged Flash Attention Varlen");
}