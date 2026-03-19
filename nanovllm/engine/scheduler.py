from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
        
            
    ########## nano vllm V1 add ##########
    def schedule_with_chunked_prefill(self) -> tuple[list[Sequence], list[Sequence], dict[int, int]]:
        
        preempt_seqs: list[Sequence] = []
        scheduled_new_seqs: list[Sequence] = []
        scheduled_running_seqs: list[Sequence] = []
        
        seq_to_new_blocks: dict[int, list] = {}
        num_scheduled_tokens: dict[int, int] = {}
        token_budget = self.max_num_batched_tokens
        
        running_seq_id = 0
        while running_seq_id < len(self.running) and token_budget > 0:
            seq = self.running[running_seq_id]  # 顺序取 seq，暂不使用任何挑选策略
            max_new_token = len(seq) - seq.num_computed_tokens
            
            max_new_token = min(token_budget, max_new_token)
            
            # Schedule newly needed KV blocks for the request.
            while True:
                new_blocks = self.block_manager.allocate_slots(
                    seq = seq,
                    num_new_tokens = max_new_token,
                    )
                
                if new_blocks is not None:
                    # The request can be scheduled.
                    break
                
                preempted_seq = self.running.pop()
                
                self.preempt(preempted_seq)
                
                preempt_seqs.append(preempted_seq)
                if preempted_seq == seq:
                    # No more request to preempt. Cannot schedule this request.
                    break
            
            if new_blocks is None:
                # Cannot schedule this request.
                break
            
            scheduled_running_seqs.append(seq)
            seq_id = seq.seq_id
            num_scheduled_tokens[seq_id] = max_new_token
            seq_to_new_blocks[seq_id] = new_blocks
            
            token_budget -= max_new_token
            
            running_seq_id += 1
        
        # 调度 waitting 队列
        if len(preempt_seqs) == 0: # 没有发生抢占再处理 waitting 队列
            while self.waiting and token_budget > 0:
                seq = self.waiting[0]  # 直接取第一个，暂不使用任何挑选策略
                max_new_token = len(seq) - seq.num_computed_tokens
                
                if seq.num_computed_tokens == 0:
                    # 计算已经被 prefix cache 的 block，这一操作没有增加 block 的 ref_count，也没有将
                    # cache 命中的 block 加入到 seq 的 block table 中，这些操作将由 
                    # self.block_manager.allocate_slots 完成
                    new_computed_blocks, num_new_local_computed_tokens = (
                            self.block_manager.get_computed_blocks(seq)
                        )
                    
                    num_computed_tokens = num_new_local_computed_tokens
                else:
                    # TODO: 按照目前设计的逻辑来看，这部分永不可能进入
                    new_computed_blocks = None
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = seq.num_computed_tokens
                    
                max_new_token = len(seq) - num_computed_tokens
                
                max_new_token = min(max_new_token, token_budget)
                
                # 1) 计算 new block 的数量
                # 2）处理 new_computed_blocks 的 ref_count 并添加到 seq.block_table 中
                # 3）分配新 block
                # 4）为新的 full block 计算 prefix hash 并添加到 hash table，处理 seq.num_cached_tokens
                new_blocks = self.block_manager.allocate_slots(
                    seq = seq,
                    num_new_tokens = max_new_token,
                    num_new_computed_tokens=num_new_local_computed_tokens,
                    new_computed_blocks=new_computed_blocks,
                )
                
                if new_blocks is None:
                    # Cannot schedule this request.
                    break
                
                self.waiting.popleft()
                self.running.append(seq)
                
                scheduled_new_seqs.append(seq)
                
                seq_id = seq.seq_id
                num_scheduled_tokens[seq_id] = max_new_token
                token_budget -= max_new_token
                seq.status = SequenceStatus.RUNNING
                
                ## 已经计算过的 token 数量，对于新的 seq 这一数量为命中 prefix cache 的 token数量
                seq.num_computed_tokens = num_computed_tokens 
                
                # if seq.num_cached_tokens == 0:
                #     seq.num_cached_tokens = num_computed_tokens
                    
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_batched_tokens
        assert token_budget >= 0
        
        return scheduled_new_seqs, scheduled_running_seqs, num_scheduled_tokens
        
    def update_from_output(
        self,
        scheduled_new_seqs: list[Sequence],
        scheduled_running_seqs: list[Sequence],
        num_scheduled_tokens: dict[int, int],
        sampled_token_ids: list[int | None]
    ):
        all_seqs = scheduled_new_seqs + scheduled_running_seqs
        for i, seq in enumerate(all_seqs):
            seq_id = seq.seq_id
            num_new_token = num_scheduled_tokens[seq_id]
            seq.num_computed_tokens += num_new_token # 添加已经计算的 token 数
            
            generated_token_id = sampled_token_ids[i]
            if not generated_token_id:
                continue
            
            # 需要添加新的 token
            seq.append_token(generated_token_id)
            if (not seq.ignore_eos and generated_token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
       