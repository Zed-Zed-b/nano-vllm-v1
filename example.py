import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    # exit()
    path = os.path.expanduser("/home/featurize/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca")
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1, kvcache_block_size=256)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {output['text']!r}")
        print("-" * 60)


if __name__ == "__main__":
    main()