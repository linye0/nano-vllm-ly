#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>
#include <mma.h>
#include <memory>
#include <unordered_map>
#include <iostream>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__device__ __forceinline__ void load_global_to_shared_bf16(
    __nv_bfloat16* s_ptr,
    const __nv_bfloat16* g_ptr,
    const int d,
    const int stride,
    const int br,
    const int D_padded,
    int cur_seqlen_q,
    int q_chunk_base_idx
) {
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    int chunks_per_row = d / 8; 
    int total_chunks = br * chunks_per_row;

    for (int i = tid; i < total_chunks; i += num_threads) {
        int row = i / chunks_per_row;
        int col = i % chunks_per_row;

        __nv_bfloat16* s_chunk_ptr = s_ptr + row * D_padded + col * 8;
        if (q_chunk_base_idx + row < cur_seqlen_q) {
            const __nv_bfloat16* g_chunk_ptr = g_ptr + row * stride + col * 8;
            *reinterpret_cast<uint4*>(s_chunk_ptr) = *reinterpret_cast<const uint4*>(g_chunk_ptr);
        } else {
            *reinterpret_cast<uint4*>(s_chunk_ptr) = make_uint4(0, 0, 0, 0);
        }
    }
}

__device__ __forceinline__ void load_global_to_shared_bf16_async(
    int stage,
    __nv_bfloat16* s_ptr,
    const __nv_bfloat16* g_ptr,
    const int d,
    const int stride,
    const int br,
    const int D_padded,
    int cur_seqlen_k,
    int logical_idx_base
) {
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    int chunks_per_row = d / 8; 
    int total_chunks = br * chunks_per_row;

    #pragma unroll
    for (int i = tid; i < total_chunks; i += num_threads) {
        int row = i / chunks_per_row;
        int col = i % chunks_per_row;

        if (logical_idx_base + row < cur_seqlen_k) {
            const __nv_bfloat16* g_chunk_ptr = reinterpret_cast<const __nv_bfloat16*>(g_ptr + row * stride + col * 8);
            __nv_bfloat16* s_chunk_ptr = reinterpret_cast<__nv_bfloat16*>(s_ptr + stage * br * D_padded + row * D_padded + col * 8);
            __pipeline_memcpy_async(s_chunk_ptr, g_chunk_ptr, 16);
        }
    }
}

__device__ __forceinline__ void apply_mask(
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>& frag0,
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>& frag1,
    int logical_k_base, int logical_q_row_base, 
    int cur_seqlen_k, int seq_offset, int tid
) {
    int lane = tid % 32;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int r = (lane / 4) + ((i % 4) / 2) * 8;
        int c = (lane % 4) * 2 + (i % 2) + (i / 4) * 8;

        int l_q = logical_q_row_base + r + seq_offset; 
        
        int global_k_0 = logical_k_base + c;
        int global_k_1 = logical_k_base + 16 + c;

        if (global_k_0 >= cur_seqlen_k || global_k_0 > l_q) frag0.x[i] = -CUDART_INF_F; 
        if (global_k_1 >= cur_seqlen_k || global_k_1 > l_q) frag1.x[i] = -CUDART_INF_F;
    }
}

template<int D, int Br = 64, int Bc = 32>
__global__ void flash_attn_prefill_kernel(
    const __nv_bfloat16* __restrict__ Q, // [num_tokens, num_heads, head_dim]
    const __nv_bfloat16* __restrict__ K_cache, // paged: [num_blocks, block_dim, num_heads, head_dim] unpaged: [num_tokens, num_heads, head_dim]
    const __nv_bfloat16* __restrict__ V_cache, // 同k
    __nv_bfloat16* __restrict__ O, // 同Q
    const int* __restrict__ cu_seqlens_q, 
    const int* __restrict__ cu_seqlens_k, 
    const int* __restrict__ block_table, 
    const float scale,
    const int num_heads,
    const int num_kv_heads,
    const int block_size, 
    const int max_blocks_per_seq
) {
    constexpr int num_d_steps = D / WMMA_K;
    int batch_id = blockIdx.z;
    int head_id = blockIdx.y; 
    int q_chunk_idx = blockIdx.x; 
    int tid = threadIdx.x;

    int wid = tid / 32;
    int row_offset_q = wid * WMMA_M;

    int kv_head_id = head_id / (num_heads / num_kv_heads);

    int q_start_offset = cu_seqlens_q[batch_id];
    int q_end_offset = cu_seqlens_q[batch_id + 1];
    int cur_seqlen_q = q_end_offset - q_start_offset;

    int kv_start_offset = cu_seqlens_k[batch_id];
    int kv_end_offset = cu_seqlens_k[batch_id + 1];
    int cur_seqlen_kv = kv_end_offset - kv_start_offset;

    if (q_chunk_idx * Br >= cur_seqlen_q) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> q_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> k_frag[2];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_frag[2];
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> p_frag[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> v_frag[2];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag[num_d_steps];

    for (int vi = 0; vi < num_d_steps; ++vi) {
        wmma::fill_fragment(o_frag[vi], 0.0f);
    }

    const int D_padded = D + 8;

    extern __shared__ char dynamic_sram[];
    __nv_bfloat16* sram = (__nv_bfloat16*)dynamic_sram;
    __nv_bfloat16* s_Q = sram;
    __nv_bfloat16* s_K = s_Q + Br * D_padded; 
    __nv_bfloat16* s_V = s_K + 2 * Bc * D_padded;
    
    // 新增：专门用于碎片布局转换的 P 矩阵中转站
    float* s_P_base = (float*)(s_V + 2 * Bc * D_padded);

    int write_stage = 0; 
    int read_stage = 0;

    float m1 = -CUDART_INF_F, m2 = -CUDART_INF_F;
    float l1 = 0.0f, l2 = 0.0f;

    __nv_bfloat16* q_ptr = const_cast<__nv_bfloat16*>(Q) + (q_start_offset + q_chunk_idx * Br) * num_heads * D + (head_id * D);
    load_global_to_shared_bf16(s_Q, q_ptr, D, num_heads * D, Br, D_padded, cur_seqlen_q, q_chunk_idx * Br);
    __syncthreads();

    size_t kv_stride = (size_t)num_kv_heads * D;
    int l_q_base = q_chunk_idx * Br + row_offset_q;

    int j = 0; 
    {
        int logical_idx = j * Bc;
        size_t p_block_offset;
        if (block_table != nullptr) {
            int kv_block_idx = logical_idx / block_size;
            int kv_offset_in_block = logical_idx % block_size;
            int physical_block_id = block_table[batch_id * max_blocks_per_seq + kv_block_idx];
            p_block_offset = (size_t)physical_block_id * block_size + kv_offset_in_block;
        } else {
            p_block_offset = kv_start_offset + logical_idx;
        }

        const __nv_bfloat16* k_ptr = K_cache + p_block_offset * kv_stride + (kv_head_id * D);
        const __nv_bfloat16* v_ptr = V_cache + p_block_offset * kv_stride + (kv_head_id * D);
        load_global_to_shared_bf16_async(write_stage, s_K, k_ptr, D, num_kv_heads * D, Bc, D_padded, cur_seqlen_kv, logical_idx);
        load_global_to_shared_bf16_async(write_stage, s_V, v_ptr, D, num_kv_heads * D, Bc, D_padded, cur_seqlen_kv, logical_idx);
        write_stage ^= 1;
        __pipeline_commit();
    }

    for (j = 1; j < (cur_seqlen_kv + Bc - 1) / Bc; ++j) {
        int logical_idx = j * Bc;
        size_t p_block_offset;
        if (block_table != nullptr) {
            int kv_block_idx = logical_idx / block_size;
            int kv_offset_in_block = logical_idx % block_size;
            int physical_block_id = block_table[batch_id * max_blocks_per_seq + kv_block_idx];
            p_block_offset = (size_t)physical_block_id * block_size + kv_offset_in_block;
        } else {
            p_block_offset = kv_start_offset + logical_idx;
        }

        const __nv_bfloat16* k_ptr = K_cache + p_block_offset * kv_stride + (kv_head_id * D);
        const __nv_bfloat16* v_ptr = V_cache + p_block_offset * kv_stride + (kv_head_id * D);
        load_global_to_shared_bf16_async(write_stage, s_K, k_ptr, D, num_kv_heads * D, Bc, D_padded, cur_seqlen_kv, logical_idx);
        load_global_to_shared_bf16_async(write_stage, s_V, v_ptr, D, num_kv_heads * D, Bc, D_padded, cur_seqlen_kv, logical_idx);
        __pipeline_commit();

        __pipeline_wait_prior(1);
        __syncthreads();

        wmma::fill_fragment(s_frag[0], 0.0f);
        wmma::fill_fragment(s_frag[1], 0.0f);

        #pragma unroll
        for (int ki = 0; ki < num_d_steps; ++ki) {
            const __nv_bfloat16* q_tile_ptr = s_Q + row_offset_q * D_padded + ki * WMMA_K;
            wmma::load_matrix_sync(q_frag, q_tile_ptr, D_padded);

            const __nv_bfloat16* k_ptr_0 = s_K + read_stage * Bc * D_padded + 0 * WMMA_N * D_padded + ki * WMMA_K;
            const __nv_bfloat16* k_ptr_1 = s_K + read_stage * Bc * D_padded + 1 * WMMA_N * D_padded + ki * WMMA_K; 
            wmma::load_matrix_sync(k_frag[0], k_ptr_0, D_padded);
            wmma::load_matrix_sync(k_frag[1], k_ptr_1, D_padded);

            wmma::mma_sync(s_frag[0], q_frag, k_frag[0], s_frag[0]);
            wmma::mma_sync(s_frag[1], q_frag, k_frag[1], s_frag[1]);
        }

        apply_mask(s_frag[0], s_frag[1], (j - 1) * Bc, l_q_base, cur_seqlen_kv, cur_seqlen_kv - cur_seqlen_q, tid);

        float m1_local = -CUDART_INF_F, m2_local = -CUDART_INF_F;

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float val0 = s_frag[0].x[i] * scale;
            float val1 = s_frag[1].x[i] * scale;

            if ((i % 4) < 2) {
                m1_local = fmaxf(m1_local, fmaxf(val0, val1));
            } else {
                m2_local = fmaxf(m2_local, fmaxf(val0, val1));
            }
        }

        #pragma unroll
        for (int mask = 2; mask > 0; mask >>= 1) {
            m1_local = fmaxf(m1_local, __shfl_xor_sync(0xffffffff, m1_local, mask));
            m2_local = fmaxf(m2_local, __shfl_xor_sync(0xffffffff, m2_local, mask));
        }

        float m1_new = fmaxf(m1, m1_local);
        float m2_new = fmaxf(m2, m2_local);

        float sum1_local = 0.0f, sum2_local = 0.0f; 

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float p0, p1;
            if ((i % 4) < 2) {
                p0 = expf(s_frag[0].x[i] * scale - m1_new);
                p1 = expf(s_frag[1].x[i] * scale - m1_new);
                sum1_local += (p0 + p1);
            } else {
                p0 = expf(s_frag[0].x[i] * scale - m2_new);
                p1 = expf(s_frag[1].x[i] * scale - m2_new);
                sum2_local += (p0 + p1);
            }
            
            // 修复关键点：将算出的概率直接塞回 accumulator fragment！
            s_frag[0].x[i] = p0;
            s_frag[1].x[i] = p1;
        }

        #pragma unroll
        for (int mask = 2; mask > 0; mask >>= 1) {
            sum1_local += __shfl_xor_sync(0xffffffff, sum1_local, mask);
            sum2_local += __shfl_xor_sync(0xffffffff, sum2_local, mask);
        }

        float scale1_o = expf(m1 - m1_new);
        float scale2_o = expf(m2 - m2_new);

        l1 = l1 * scale1_o + sum1_local;
        l2 = l2 * scale2_o + sum2_local;
        m1 = m1_new;
        m2 = m2_new;

        for (int vi = 0; vi < num_d_steps; ++vi) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                if ((i % 4) < 2) o_frag[vi].x[i] *= scale1_o;
                else             o_frag[vi].x[i] *= scale2_o;
            }
        }

        // 核心中转站：通过 Shared Memory 重排 P 矩阵的布局
        float* warp_s_P_f = s_P_base + wid * 512;
        // 1. 从 accumulator 写出到 Shared Memory (自动转换为标准的行主序)
        wmma::store_matrix_sync(warp_s_P_f,       s_frag[0], 16, wmma::mem_row_major);
        wmma::store_matrix_sync(warp_s_P_f + 256, s_frag[1], 16, wmma::mem_row_major);
        __syncwarp();

        // 2. Warp 内部并发将 float 转为 bf16
        __nv_bfloat16* warp_s_P_bf16 = (__nv_bfloat16*)warp_s_P_f;
        int lane = tid % 32;
        for (int i = lane; i < 512; i += 32) {
            warp_s_P_bf16[i] = __float2bfloat16(warp_s_P_f[i]);
        }
        __syncwarp();

        // 3. 作为正确的 matrix_a 类型读回来！
        wmma::load_matrix_sync(p_frag[0], warp_s_P_bf16,       16);
        wmma::load_matrix_sync(p_frag[1], warp_s_P_bf16 + 256, 16);

        const __nv_bfloat16* cur_s_V = s_V + read_stage * Bc * D_padded;
        for (int vi = 0; vi < num_d_steps; ++vi) {
            wmma::load_matrix_sync(v_frag[0], cur_s_V + 0 * WMMA_N * D_padded + vi * WMMA_K, D_padded);
            wmma::load_matrix_sync(v_frag[1], cur_s_V + 1 * WMMA_N * D_padded + vi * WMMA_K, D_padded);

            wmma::mma_sync(o_frag[vi], p_frag[0], v_frag[0], o_frag[vi]);
            wmma::mma_sync(o_frag[vi], p_frag[1], v_frag[1], o_frag[vi]);
        }
        
        write_stage ^= 1;
        read_stage ^= 1;
        __syncthreads();
    }

    __pipeline_wait_prior(0);
    __syncthreads();

    wmma::fill_fragment(s_frag[0], 0.0f);
    wmma::fill_fragment(s_frag[1], 0.0f);
    
    #pragma unroll
    for (int ki = 0; ki < num_d_steps; ++ki) {
        const __nv_bfloat16* q_tile_ptr = s_Q + row_offset_q * D_padded + ki * WMMA_K;
        wmma::load_matrix_sync(q_frag, q_tile_ptr, D_padded);

        const __nv_bfloat16* k_ptr_0 = s_K + read_stage * Bc * D_padded + 0 * WMMA_N * D_padded + ki * WMMA_K;
        const __nv_bfloat16* k_ptr_1 = s_K + read_stage * Bc * D_padded + 1 * WMMA_N * D_padded + ki * WMMA_K;
        wmma::load_matrix_sync(k_frag[0], k_ptr_0, D_padded);
        wmma::load_matrix_sync(k_frag[1], k_ptr_1, D_padded);

        wmma::mma_sync(s_frag[0], q_frag, k_frag[0], s_frag[0]);
        wmma::mma_sync(s_frag[1], q_frag, k_frag[1], s_frag[1]);
    }

    apply_mask(s_frag[0], s_frag[1], (j - 1) * Bc, l_q_base, cur_seqlen_kv, cur_seqlen_kv - cur_seqlen_q, tid);

    float m1_local = -CUDART_INF_F, m2_local = -CUDART_INF_F;

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
         float val0 = s_frag[0].x[i] * scale;
         float val1 = s_frag[1].x[i] * scale;

         if ((i % 4) < 2) {
            m1_local = fmaxf(m1_local, fmaxf(val0, val1));
         } else {
            m2_local = fmaxf(m2_local, fmaxf(val0, val1));
         }
    }

    #pragma unroll
    for (int mask = 2; mask > 0; mask >>= 1) {
        m1_local = fmaxf(m1_local, __shfl_xor_sync(0xffffffff, m1_local, mask));
        m2_local = fmaxf(m2_local, __shfl_xor_sync(0xffffffff, m2_local, mask));
    }

    float m1_new = fmaxf(m1, m1_local);
    float m2_new = fmaxf(m2, m2_local);

    float sum1_local = 0.0f, sum2_local = 0.0f;

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float p0, p1;
        if ((i % 4) < 2) {
            p0 = expf(s_frag[0].x[i] * scale - m1_new);
            p1 = expf(s_frag[1].x[i] * scale - m1_new);
            sum1_local += (p0 + p1);
        } else {
            p0 = expf(s_frag[0].x[i] * scale - m2_new);
            p1 = expf(s_frag[1].x[i] * scale - m2_new);
            sum2_local += (p0 + p1);
        }
        
        s_frag[0].x[i] = p0;
        s_frag[1].x[i] = p1;
    }

    #pragma unroll
    for (int mask = 2; mask > 0; mask >>= 1) {
        sum1_local += __shfl_xor_sync(0xffffffff, sum1_local, mask);
        sum2_local += __shfl_xor_sync(0xffffffff, sum2_local, mask);
    }

    float scale1_o = expf(m1 - m1_new);
    float scale2_o = expf(m2 - m2_new);

    l1 = l1 * scale1_o + sum1_local;
    l2 = l2 * scale2_o + sum2_local;
    m1 = m1_new;
    m2 = m2_new;

    for(int vi = 0; vi < num_d_steps; ++vi) {
        #pragma unroll
        for(int i=0; i<8; ++i) {
            if ((i % 4) < 2) o_frag[vi].x[i] *= scale1_o;
            else             o_frag[vi].x[i] *= scale2_o;
        }
    }

    float* warp_s_P_f = s_P_base + wid * 512;
    wmma::store_matrix_sync(warp_s_P_f,       s_frag[0], 16, wmma::mem_row_major);
    wmma::store_matrix_sync(warp_s_P_f + 256, s_frag[1], 16, wmma::mem_row_major);
    __syncwarp();

    __nv_bfloat16* warp_s_P_bf16 = (__nv_bfloat16*)warp_s_P_f;
    int lane = tid % 32;
    for (int i = lane; i < 512; i += 32) {
        warp_s_P_bf16[i] = __float2bfloat16(warp_s_P_f[i]);
    }
    __syncwarp();

    wmma::load_matrix_sync(p_frag[0], warp_s_P_bf16,       16);
    wmma::load_matrix_sync(p_frag[1], warp_s_P_bf16 + 256, 16);

    const __nv_bfloat16* cur_s_V = s_V + read_stage * Bc * D_padded;
    for (int vi = 0; vi < num_d_steps; ++vi) {
        wmma::load_matrix_sync(v_frag[0], cur_s_V + 0 * WMMA_N * D_padded + vi * WMMA_K, D_padded);
        wmma::load_matrix_sync(v_frag[1], cur_s_V + 1 * WMMA_N * D_padded + vi * WMMA_K, D_padded);

        wmma::mma_sync(o_frag[vi], p_frag[0], v_frag[0], o_frag[vi]);
        wmma::mma_sync(o_frag[vi], p_frag[1], v_frag[1], o_frag[vi]);
    }

    for(int vi = 0; vi < num_d_steps; ++vi) {
        #pragma unroll
        for(int i = 0; i < 8; ++i) {
            if ((i % 4) < 2) {
                o_frag[vi].x[i] /= l1;
            } else {
                o_frag[vi].x[i] /= l2;
            }
        }
    }

    int global_row_idx = q_start_offset + q_chunk_idx * Br + row_offset_q;
    float* s_out_trampoline = (float*)s_Q + wid * 256;

    for (int vi = 0; vi < num_d_steps; ++vi) {
        wmma::store_matrix_sync(s_out_trampoline, o_frag[vi], WMMA_N, wmma::mem_row_major);
        __syncwarp(); 

        int lane_id = tid % 32;
        
        for (int e = 0; e < 8; ++e) {
            int elem_idx = lane_id * 8 + e;
            int r = elem_idx / 16;
            int c = elem_idx % 16;

            if (global_row_idx + r < q_end_offset) {
                float val = s_out_trampoline[elem_idx];
                size_t g_idx = (size_t)(global_row_idx + r) * num_heads * D + (head_id * D) + vi * WMMA_K + c;
                O[g_idx] = __float2bfloat16(val);
            }
        }
        __syncwarp(); 
    }
}

template<int D, int partition_size = 128>
__global__ void flash_attn_decode_partial_kernel(
    const __nv_bfloat16* __restrict__ q,           // [batch, 1, n_heads, head_dim]
    const __nv_bfloat16* __restrict__ k_cache,     // [num_blocks, block_size, n_heads_kv, head_dim]
    const __nv_bfloat16* __restrict__ v_cache,     // [num_blocks, block_size, n_heads_kv, head_dim]
    const int* __restrict__ block_table,           // [batch, max_num_blocks]
    const int* __restrict__ context_lens,          // [batch]
    float* __restrict__ tmp_out,                   // [batch, n_heads, num_splits, d]
    float* __restrict__ tmp_lse,                   // [batch, n_heads, num_splits, 2] (m 和 l)
    int n_heads, int n_heads_kv, 
    int block_size, int max_blocks_per_seq, 
    float scale
) {
    constexpr int head_dim = D;
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int split_idx = blockIdx.z;
    int tidx = threadIdx.x;
    int kv_head_idx = head_idx / (n_heads / n_heads_kv);
    int num_splits = gridDim.z;

    // 越界保护
    int seq_len = context_lens[batch_idx];
    int start_token = split_idx * partition_size;

    int valid_tokens_in_split = max(0, min(partition_size, seq_len - start_token));

    extern __shared__ char dynamic_sram[];
    __nv_bfloat16* sram = (__nv_bfloat16*)dynamic_sram;
    __nv_bfloat16* s_Q = sram;

    // 因为这边的K和V是完全不复用的，所以只需要把Q加载到sram里面就可以了
    const __nv_bfloat16* q_ptr = q + batch_idx * n_heads * head_dim + head_idx * head_dim;
    #pragma unroll
    for (int i = tidx * 8; i < head_dim; i += blockDim.x * 8) {
        // 把8个bfloat16视作一个uint4进行合并读取
        *reinterpret_cast<uint4*>(&s_Q[i]) = *reinterpret_cast<const uint4*>(&q_ptr[i]);
    }
    __syncthreads();

    int kv_stride =  n_heads_kv * head_dim;

    float m_i = -1e20f;
    float l_i = 0.0f;
    float o_reg[head_dim] = {0.0f};

    for (int i = tidx; i < valid_tokens_in_split; i += blockDim.x) {
        // 对于每个thread，分别计算起始位置
        int global_token_idx = start_token + i;
        int logical_block_idx = global_token_idx / block_size;
        int offset_in_block = global_token_idx % block_size;

        int physical_block_idx = block_table[batch_idx * max_blocks_per_seq + logical_block_idx];

        int kv_start_offset = physical_block_idx * block_size * kv_stride
                            + offset_in_block * kv_stride
                            + kv_head_idx * head_dim;

        const __nv_bfloat16* k_cur_ptr = k_cache + kv_start_offset;
        const __nv_bfloat16* v_cur_ptr = v_cache + kv_start_offset;

        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < head_dim; k += 8) {
            const float4 k_raw = *(const float4*)(&k_cur_ptr[k]);
            const __nv_bfloat162* k_bf16x2 = reinterpret_cast<const __nv_bfloat162*>(&k_raw);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 k_f2 = __bfloat1622float2(k_bf16x2[j]);
                sum += k_f2.x * __bfloat162float(s_Q[k + j * 2 + 0]);
                sum += k_f2.y * __bfloat162float(s_Q[k + j * 2 + 1]);
            }
        }
        sum *= scale;

        // Online softmax 更新
        float m_prev = m_i;
        m_i = fmaxf(m_prev, sum);
        float alpha = expf(m_prev - m_i);
        float beta = expf(sum - m_i);
        l_i = l_i * alpha + beta;

        #pragma unroll
        for (int v = 0; v < head_dim; v += 8) {
            const float4 v_raw = *(const float4*)(&v_cur_ptr[v]);
            const __nv_bfloat162* v_bf16x2 = reinterpret_cast<const __nv_bfloat162*>(&v_raw);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 v_f2 = __bfloat1622float2(v_bf16x2[j]);
                o_reg[v + j * 2 + 0] = o_reg[v + j * 2 + 0] * alpha + beta * v_f2.x;
                o_reg[v + j * 2 + 1] = o_reg[v + j * 2 + 1] * alpha + beta * v_f2.y;
            }
        }
    }

    // 块内归约
    // 第一步: warp shuffle规约(32线程合并)
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float m_other = __shfl_xor_sync(0xffffffff, m_i, offset);
        float l_other = __shfl_xor_sync(0xffffffff, l_i, offset);

        float m_max = fmaxf(m_i, m_other);
        float a = expf(m_i - m_max);
        float b = expf(m_other - m_max);

        m_i = m_max;
        l_i = l_i * a + l_other * b;
        #pragma unroll
        for (int k = 0; k < head_dim; ++k) {
            float o_other = __shfl_xor_sync(0xffffffff, o_reg[k], offset);
            o_reg[k] = o_reg[k] * a + o_other * b;
        }
    }

    // 跨warp规约(使用Shared Mem交换各个warp领头线程的结果)
    __shared__ float s_m[32];
    __shared__ float s_l[32];
    __shared__ float s_o[32][head_dim];

    int lane = tidx % 32;
    int wid = tidx / 32;
    int num_warps = blockDim.x / 32;

    if (lane == 0) {
        s_m[wid] = m_i;
        s_l[wid] = l_i;
        #pragma unroll
        for (int k = 0; k < head_dim; ++k) s_o[wid][k] = o_reg[k];
    }

    __syncthreads();

    // 由tx=0线程对四个warp的结果进行合并
    if (tidx == 0) {
        float m_block = s_m[0];
        float l_block = s_l[0];
        float o_block[head_dim];
        #pragma unroll
        for (int k = 0; k < head_dim; ++k) o_block[k] = s_o[0][k];

        for (int w = 1; w < num_warps; ++w) {
            float m_w = s_m[w];
            float l_w = s_l[w];
            float m_max = fmaxf(m_block, m_w);
            float a = expf(m_block - m_max);
            float b = expf(m_w - m_max);

            l_block = l_block * a + l_w * b;
            m_block = m_max;
            #pragma unroll
            for (int k = 0; k < head_dim; ++k) o_block[k] = o_block[k] * a + s_o[w][k] * b;
        }

        // 写回workspace，供reduction phase使用
        int base_offset = (batch_idx * n_heads + head_idx) * num_splits;
        int lse_offset = (base_offset + split_idx) * 2;
        int out_offset = (base_offset + split_idx) * head_dim;
        
        tmp_lse[lse_offset + 0] = m_block;
        tmp_lse[lse_offset + 1] = l_block;
        #pragma unroll
        for (int k = 0; k < head_dim; ++k) tmp_out[out_offset + k] = o_block[k];
    }
}

template<int D>
__global__ void flash_attn_decode_reduce_kernel(
    const float* __restrict__ tmp_out, // [batch, n_heads, num_splits, head_dim]
    const float* __restrict__ tmp_lse, // [batch, n_heads, num_splits, 2]
    __nv_bfloat16* __restrict__ out, // [batch, 1, n_heads, head_dim]
    int num_splits,
    int n_heads
) {
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tidx = threadIdx.x;

    if (tidx >= D) return;

    const float* lse_ptr = tmp_lse + (batch_idx * n_heads + head_idx) * num_splits * 2;
    const float* out_ptr = tmp_out + (batch_idx * n_heads + head_idx) * num_splits * D;

    float m_max = -1e20f;
    for (int s = 0; s < num_splits; ++s) {
        m_max = fmaxf(m_max, lse_ptr[s * 2 + 0]);
    }

    float acc_v = 0.0f;
    float sum_l = 0.0f;
    for (int s = 0; s < num_splits; ++s) {
        float m_s = lse_ptr[s * 2 + 0];
        float l_s = lse_ptr[s * 2 + 1];

        float w = expf(m_s - m_max);
        acc_v += out_ptr[s * D + tidx] * w;
        sum_l += l_s * w;
    }

    float final_v = acc_v / sum_l;
    out[(batch_idx * n_heads + head_idx) * D + tidx] = __float2bfloat16(final_v);
}