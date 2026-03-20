import json
import os

import numpy as np
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple

rng = np.random.default_rng(42)
def generate_random_prompt(prompt_len: int, rng):
    return rng.integers(0, 5000, size=prompt_len).tolist()

def load_ttft_testdata(
    data_file_path: str,
    temperature: float = 0.6,
    ignore_eos: bool = True,                       
) -> Tuple[List[List[int]], List[SamplingParams]]:
    
    with open(data_file_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
        
    reqs = payload["requests"]
    prompts: List[List[int]] = []
    sampling_params_list: List[SamplingParams] = []
    
    for req in reqs:
        prompt_len = int(req["prompt_len"])
        max_tokens = int(req["max_tokens"])
        prompt_token_id = int(req.get("prompt_token_id", payload.get("prompt_token_id_default", 1)))
        # 全用同一个 token id，控制长度即可
        # prompts.append([prompt_token_id] * prompt_len)
        prompts.append(generate_random_prompt(prompt_len, rng))
        sampling_params_list.append(
            SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                ignore_eos=ignore_eos,
            )
        )
    return prompts, sampling_params_list

def main():
    path = os.path.expanduser("/home/featurize/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca")
    testdata_path = "/home/featurize/work/mydata/nano-vllm/nano-vllm/data/ttft_testdata_long_prompt.json"
    
    prompts, sampling_params_list = load_ttft_testdata(testdata_path)
    
    llm = LLM(
        path,
        enforce_eager=True,
        tensor_parallel_size=1,
        kvcache_block_size=256,
        max_model_len=1024,
        max_num_batched_tokens=8192,
    )

    outputs = llm.generate(prompts, sampling_params_list)

    # for prompt_tokens, output in zip(prompts, outputs):
    #     print(f"Prompt len: {len(prompt_tokens)}")
    #     print(f"Output:     {output['text']!r}")
    #     print("-" * 60)


if __name__ == "__main__":
    main()
