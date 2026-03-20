from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        
        # nano-vllm V1 add
        self.num_computed_tokens = 0

        # Profiling (TTFT): record request arrival time and first completion token time.
        # These fields are only used for profiling and should not affect scheduling.
        self._arrival_time: float | None = None
        self._first_completion_time: float | None = None
        self._ttft: float | None = None

        # TPOT (Time Per OutputToken): store deltas between consecutive completion tokens.
        # By definition, TPOT excludes the first completion token, so we only append deltas
        # once we have a previous completion timestamp.
        self._last_completion_time: float | None = None
        self._tpot_deltas: list[float] = []

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        return (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.block_table,
            self._arrival_time,
            self._first_completion_time,
            self._ttft,
            self._last_completion_time,
            self._tpot_deltas,
            self.token_ids if self.num_completion_tokens == 0 else self.last_token,
        )

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[0:4]
        self._arrival_time, self._first_completion_time, self._ttft = state[4:7]
        self._last_completion_time = state[7]
        self._tpot_deltas = state[8]
        token_or_last = state[9]
        if self.num_completion_tokens == 0:
            self.token_ids = token_or_last
        else:
            self.last_token = token_or_last
