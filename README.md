# Nano-VLLM-ly (Custom Edition)

这是一个基于[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)魔改的轻量级大模型推理引擎。

本项目的核心贡献在于实现并集成了一个手写的、基于 CUDA WMMA (Tensor Core) 的高性能 BFloat16 Prefill 注意力算子。通过这个算子，成功打通了从底层硬件寄存器布局到高层 Paged Attention 推理框架的全链路。

## 快速开始

### 1. 编译安装

原生nano-vllm的配置请参考[原repo](https://github.com/GeeeekExplorer/nano-vllm).

要使用自定义的kernel，首先需要编译并安装 CUDA 扩展包：

```bash
cd nanovllm/custom && python setup.py instal
```

### 2. 运行example

```bash
python example.py --custom_kernel
```

### 3. 运行benchmark

```bash
python bench.py --custom_kernel
```

**测试配置：**

* **硬件：** RTX 3060 Laptop (6GB专用显存)
* **模型：** Qwen3-0.6B
* **总请求数：** 256 条序列
* **输入长度：** 随机采样，范围 100–1024 Tokens
* **输出长度：** 随机采样，范围 100–1024 Tokens

**性能测试结果：**

| 推理引擎 | 输出 Tokens | 耗时 (s) | 吞吐量 (tokens/s) |
| :--- | :--- | :--- | :--- |
| Nano-vLLM (原生 Triton 基线) | 133,966 | 124.11 | 1079.41 |
| **Nano-vLLM-ly (自定义 CUDA kernel)** | **133,966** | **66.57** | **2012.33** |

**性能归因分析：**

通过将原生的 Triton 实现替换为深度定制的 CUDA C++ 算子，在相同硬件下，端到端 Decoding 阶段的吞吐量提升了约 86.4%。这一巨大的性能飞跃主要归功于以下底层的极致榨取：

- **128-bit 向量化访存 (float4):** 在 Paged Attention 抓取跳跃的 KV Cache 物理块时，强行对齐内存地址并打满总线带宽，消除了 Triton 编译器保守的非合并访存劣势。

- **指令级控制流优化：** 通过纯 C++ 硬编码循环与寻址公式，彻底清除了 Triton 在处理动态块映射 (Block Table) 时产生的冗余指针追踪与高频整数除法/取模开销。

- **细粒度 Warp 级规约：** 在 Online Softmax 阶段，使用硬件级的 __shfl_xor_sync 原语配合极简的 Shared Memory 布局，实现了低延迟的块内与跨 Warp 规约计算。

