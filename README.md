# Nano-vLLM-V1

在 [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 基础上做 **vLLM V1 风格** 的重构，目前已完成 **Chunked Prefill** 支持：在统一调度下将长 prompt 的 prefill 拆成多段执行，并与 decode 更灵活地共享 batch token 预算，从而改善典型场景下的 **首 token 延迟（TTFT）** 指标。


## Feature

**Chunked Prefill**：`LLMEngine`、`ModelRunner`、`Scheduler` 与 `BlockManager` 对 chunked prefill 逻辑均有对应实现；V1 新增核心逻辑在代码中以 `# nano-vllm V1 add` 标识。

### Next Step
对整体流程进行 Profile 分析，定位延迟瓶颈并加以改进。

## Installation

```bash
pip install -e .
```

## Quick Start

用法见 `example.py`：

```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

测试脚本：`profile/profile_ttft_test.py`。测试数据及 prompt 长度分布说明见 `data/ttft_testdata.json`。

**测试环境：**

| 名称 | 配置 |
| --- | --- |
| Hardware | NVIDIA RTX A4000 × 1 |
| Model | Qwen3-0.6B |
| Requests | 256 条序列 |



### 实验结果

| Method | TTFT mean (ms) | TTFT p50 (ms) | TTFT p95 (ms) | TTFT p99 (ms) |
| --- | ---: | ---: | ---: | ---: |
| Nano-vLLM | 1769 | 1771 | 2092 | 2092 |
| **Nano-vLLM-V1** | **1504** | **1392** | **1846** | 2619 |

相比于 Nano-vLLM，V1 在 **TTFT 均值约降低 15%**，**p50 / p95** 则更低，说明在相同负载下多数请求更快拿到首 token。
