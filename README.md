<p align="center">
  <img src="fig/image-2.png" width="30%" />
</p>

# Nano-VLLM-ly (Custom Edition)

这是一个基于[nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)魔改的轻量级大模型推理引擎。

本项目的核心贡献在于:

1. 实现并集成了一个手写的、基于 CUDA WMMA (Tensor Core) 的高性能 BFloat16 Prefill 注意力算子。通过这个算子，成功打通了从底层硬件寄存器布局到高层 Paged Attention 推理框架的全链路。

2. 集成了 Chunked Prefill 策略，改进了调度器的调度逻辑，使得突发长文本场景对用户的体验影响更小。

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

## 魔改功能性能测试
### 1. Custom Flash-attention Kernel 和原生 Kernel 的性能对比

使用bench.py进行测试：

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

**结果分析：**

通过将原生的 Triton 实现替换为深度定制的 CUDA C++ 算子，在相同硬件下，端到端 Decoding 阶段的吞吐量提升了约 86.4%。

### 2. Chunked Prefill 和原生调度策略的性能对比

采用test_latency.py进行测试：

```python
# 运行原生策略
python test_latency.py
# 运行chunked_prefill
python test_latency.py --chunked_prefill
```

本节通过模拟“突发长文本请求”场景，测试 nano-vllm 在不同调度策略下的引擎相应延迟，验证chunked prefill策略系统 SLA 的保护能力:

- 用户 A (延迟敏感型)：发送短请求（30 tokens），模拟连续对话，系统处于 Decode (生成) 阶段。

- 用户 B (吞吐密集型)：在用户 A 正常吐字时，突然注入一个 14,001 Tokens 的长文本请求，模拟长文档分析。

- 对比指标：逐步记录引擎 step() 函数的物理执行时间（ms）。

结果如下所示：

![chunked prefill](fig/image.png)

在第 4 步长文本请求注入时，Legacy Prefill 产生了高达 1622.8ms 的延迟峰值；而 Chunked Prefill 将该峰值压制在了 1301.2ms。

虽然在对数坐标下两者看似接近，但物理时间上 Legacy 模式的阻塞感明显更强。更重要的是，Legacy 模式在处理完大块 Prefill 后，后续步骤出现了明显的延迟波动，而 Chunked 模式则表现得极为平滑。这证明了算力切片有效地将计算压力“揉碎”到了多个时间片中。