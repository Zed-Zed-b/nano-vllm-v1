# Nano-vLLM-V1

对 Nano-vLLM 实现 vLLM V1 风格的重构。

## Feature

**Chunked Prefill Support**：实现了 EngineCore, ModelRunner, Scheduler 和 BlockManager 对 Chunked Prefill 逻辑的全面支持，所有 V1 新增核心逻辑均以 `# nano-vllm V1 add` 标识。


## Installation

```bash
pip install -e .
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```