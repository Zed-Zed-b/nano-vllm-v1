import atexit
from dataclasses import fields
import math
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        Sequence.block_size = config.kvcache_block_size
        # For profiling metrics (e.g. TTFT).
        self._all_seqs: list[Sequence] = []
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        # TTFT 起点：请求进入系统的时间
        seq._arrival_time = perf_counter()
        self._all_seqs.append(seq)
        self.scheduler.add(seq)

    def is_finished(self):
        return self.scheduler.is_finished()
    
        
    ########## nano vllm V1 add ##########
    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        self._all_seqs = []
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        # 进入 busy loop，持续处理
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step_with_chunked_prefill()
            if use_tqdm:
                throughput = num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Throughput": f"{int(throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        
        # 按照 seq id 排序输出
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()

        # TTFT profiling summary (printed at the end).
        # TTFT is measured per request: from add_request() time to first generated completion token.
        ttfts = [seq._ttft for seq in self._all_seqs if seq._ttft is not None]
        if ttfts:
            ttfts_sorted = sorted(ttfts)
            n = len(ttfts_sorted)
            def pct_nearest_rank(p: float) -> float:
                # nearest-rank percentile
                idx = int(math.ceil(p * n)) - 1
                idx = max(0, min(idx, n - 1))
                return ttfts_sorted[idx]
            ttft_ms = [x * 1000.0 for x in ttfts]
            mean_ms = sum(ttft_ms) / n
            stats = {
                "TTFT_ms_count": n,
                "TTFT_ms_mean": mean_ms,
                "TTFT_ms_p50": pct_nearest_rank(0.50) * 1000.0,
                "TTFT_ms_p95": pct_nearest_rank(0.95) * 1000.0,
                "TTFT_ms_p99": pct_nearest_rank(0.99) * 1000.0,
            }
            print(stats)
        else:
            print({"TTFT_ms_count": 0, "TTFT_ms_note": "No TTFT values recorded"})

        # TPOT profiling summary (printed at the end).
        # TPOT is measured per request token-to-token interval (excluding the first token).
        tpot_deltas = []
        for seq in self._all_seqs:
            tpot_deltas.extend(seq._tpot_deltas)
        if tpot_deltas:
            tpot_ms_sorted = sorted(x * 1000.0 for x in tpot_deltas)
            n = len(tpot_ms_sorted)

            def pct_nearest_rank_ms(p: float) -> float:
                idx = int(math.ceil(p * n)) - 1
                idx = max(0, min(idx, n - 1))
                return tpot_ms_sorted[idx]

            tpot_mean_ms = sum(tpot_ms_sorted) / n
            stats = {
                "TPOT_ms_count": n,
                "TPOT_ms_mean": tpot_mean_ms,
                "TPOT_ms_p50": pct_nearest_rank_ms(0.50),
                "TPOT_ms_p95": pct_nearest_rank_ms(0.95),
                "TPOT_ms_p99": pct_nearest_rank_ms(0.99),
            }
            print(stats)
        else:
            print({"TPOT_ms_count": 0, "TPOT_ms_note": "No TPOT values recorded"})

        return outputs
    
    def filter_token_ids(self, new_seqs: list[Sequence], 
                        running_seqs: list[Sequence],
                        num_scheduled_tokens: dict[int, int],
                        token_ids: list[int]) -> list[int]:
        """
        For requests that have not completed prefill, set the new token generated to None.
        """
        all_seqs = running_seqs + new_seqs 
        for i, seq in enumerate(all_seqs):
            seq_id = seq.seq_id
            num_new_token = num_scheduled_tokens[seq_id]
            if seq.num_computed_tokens + num_new_token < len(seq): # 未完成 prefill，不添加新的 token
                token_ids[i] = None
        
        return token_ids
            
    def step_with_chunked_prefill(self):
        scheduled_new_seqs, scheduled_running_seqs, num_scheduled_tokens = (
                self.scheduler.schedule_with_chunked_prefill()
            )
        
        token_ids = self.model_runner.call('run_chunked_prefill', 
                                           scheduled_new_seqs, 
                                           scheduled_running_seqs,
                                           num_scheduled_tokens)
        
        assert len(scheduled_new_seqs) + len(scheduled_running_seqs) == len(token_ids), \
        "The number of newly generated tokens should be equal to the number of sequences scheduled"
        
        token_ids = self.filter_token_ids(scheduled_new_seqs, 
                                          scheduled_running_seqs,
                                          num_scheduled_tokens, 
                                          token_ids)
        
        self.scheduler.update_from_output(scheduled_new_seqs, 
                                          scheduled_running_seqs,
                                          num_scheduled_tokens, 
                                          token_ids)
        
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in scheduled_running_seqs + scheduled_new_seqs if seq.is_finished]
        num_tokens = sum(num_scheduled_tokens.values())
        return outputs, num_tokens
