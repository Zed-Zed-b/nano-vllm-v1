import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    # exit()
    path = os.path.expanduser("/home/featurize/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca")
    # Profiling config: make chunked prefill easier to trigger.
    llm = LLM(
        path,
        enforce_eager=True,
        tensor_parallel_size=1,
        kvcache_block_size=256,
        max_model_len=1024,
        max_num_batched_tokens=2048,
    )

    # Use token-id prompts to avoid tokenizer overhead.
    # Also set ignore_eos=True to keep the run length controlled.
    sampling_params = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=32)

    prompt_lens = [1024, 800, 900, 600]
    token_id_for_prompt = 1
    prompts = [[token_id_for_prompt] * L for L in prompt_lens]

    outputs = llm.generate(prompts, sampling_params)

    for prompt_tokens, output in zip(prompts, outputs):
        print(f"Prompt len: {len(prompt_tokens)}")
        print(f"Output:     {output['text']!r}")
        print("-" * 60)


if __name__ == "__main__":
    main()
