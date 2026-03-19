from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []

########## nano-vllm V1 add ##########
class BlockHashToBlockMap:
    def __init__(self):
        self._cache: dict[
            int, Block | dict[int, Block]
        ] = {}
    
    def get_one_block(self, key: int) -> Block | None:
        value = self._cache.get(key, None)
        if value is None:
            return None
        elif isinstance(value, Block):
            return value
        elif isinstance(value, dict):
            return next(iter(value.values()))
        else:
            raise AssertionError(f"Invalid KV cache block type {type(value)}")
        
    def insert(self, key: int, block: Block):
        value = self._cache.get(key, None)
        if value is None:
            self._cache[key] = block
        elif isinstance(value, Block):
            self._cache[key] = {value.block_id : value, block.block_id : block}
        elif isinstance(value, dict):
            value[block.block_id] = block
        else:
            raise AssertionError(f"Invalid KV cache block type {type(value)}")

    def pop(self, key: int, block_id: int) -> Block | None:
        value = self._cache.pop(key, None)
        if value is None:
            return None
        elif isinstance(value, Block):
            if value.block_id == block_id:
                return value
            self._cache[key] = value
            return None
        elif isinstance(value, dict):
            block = value.pop(block_id, None)
            if len(value) > 0:
                self._cache[key] = value
            return block
        else:
            raise AssertionError(f"Invalid KV cache block type {type(value)}")
            
        
class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # self.hash_to_block_id: dict[int, int] = dict() # TODO: change all self.hash_to_block_id to self.hash_to_block 
        self.hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]
    
    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.hash_to_block.pop(self.blocks[block_id].hash, block_id)
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.num_computed_tokens = 0
        seq.block_table.clear()
    
    ########## nano-vllm V1 add ##########
    def cache_full_blocks(self, seq: Sequence, num_full_blocks: int):
        block_table = seq.block_table
        num_cached_blocks = seq.num_cached_blocks
        
        # block_ids_need_cache = block_table[num_cached_blocks:num_full_blocks]
        
        prefix_hash = self.blocks[block_table[num_cached_blocks - 1]].hash if num_cached_blocks > 0 else -1
        for i in range(num_cached_blocks, num_full_blocks):
            # 取出当前未进行 cache 的 full block
            block_id = block_table[i]
            block: Block = self.blocks[block_id]
            token_ids = seq.block(i)
            
            # 计算新 hash，更新 block（包括 hash 和存储的 token ids），更新 hash dict
            h = self.compute_hash(token_ids=token_ids, prefix=prefix_hash)
            block.update(h, token_ids)
            self.hash_to_block.insert(h, block)
            # self.hash_to_block_id[h] = block_id
            prefix_hash = h        
        
    def cache_blocks(self, seq: Sequence, num_tokens: int):
        """
        Cache the blocks for the sequence.

        Args:
            seq: The sequence.
            num_tokens: The total number of tokens that need to be cached
                (including tokens that are already cached).
        """
        
        num_full_blocks = num_tokens // self.block_size
        if seq.num_cached_blocks >= num_full_blocks:
            return
        
        # 计算 full block 的 cache
        self.cache_full_blocks(
            seq=seq,
            num_full_blocks=num_full_blocks,
        )
        
        # 更新属性
        seq.num_cached_tokens = num_full_blocks * self.block_size
        
    def allocate_new_computed_blocks(self, seq: Sequence,
                                     new_computed_blocks: list[int],
                                     ):
        """
        ## Maybe TODO: Touch the computed blocks to make sure they won't be evicted.

        Allocate new computed blocks for the request.
        
        Args:
            seq (Sequence): The processing sequence.
            new_computed_blocks (list): The cached blocks for the new computed tokens.
        """
        # A new sequence
        block_table = seq.block_table
        assert len(block_table) == 0
        
        # All cached hits (including skipped nulls) are already cached; mark
        # them so cache_blocks() will not try to re-cache blocks that already
        # have a block_hash set.
        for block_id in new_computed_blocks:
            self.blocks[block_id].ref_count += 1
            block_table.append(block_id)
            
        seq.num_cached_tokens += len(new_computed_blocks) * self.block_size
        
    def allocate_new_blocks(self, seq: Sequence,
                            num_blocks_to_allocate: int):
        """
        Allocate new blocks for the request.

        Args:
            seq: The processing sequence.
            num_blocks_to_allocate: The total number of blocks that need allocate.
        
        Return:
            new allocated block ids
        """
        
        new_allocated_blocks = []
        for _ in range(num_blocks_to_allocate):
            block_id = self.free_block_ids[0]
            new_allocated_blocks.append(block_id)
            self._allocate_block(block_id)
            seq.block_table.append(block_id)
        
        return new_allocated_blocks
                 
    def get_num_blocks_to_allocate(self, seq: Sequence, 
                                   num_tokens: int,
                                   new_computed_blocks: list[int]):
        required_num_blocks = (num_tokens + self.block_size - 1) // self.block_size
        
        ## <comp> 部分 + <new_comp> 部分
        num_seq_blocks = len(seq.block_table) + len(new_computed_blocks) # has hitted prefix cache blocks + just hitted cache blocks
        
        return max(required_num_blocks - num_seq_blocks, 0)
        
    def allocate_slots(self, seq: Sequence, 
                       num_new_tokens: int,
                       num_new_computed_tokens: int = 0,
                       new_computed_blocks: list[int] = None) -> list[int]:
        """Add slots for a request with new tokens to append.

        Args:
            request: The request to allocate slots.
            num_new_tokens: The number of new tokens to be allocated and computed.
            num_new_computed_tokens: The number of new computed tokens just
                hitting the prefix caching, excluding external tokens.
            new_computed_blocks: The cached blocks for the above new computed
                tokens.
        Blocks layout:
        ```
        ----------------------------------------------------------------------
        | < comp > | < new_comp > |     < new >      |
        ----------------------------------------------------------------------
                                  |    < to be       |
                                  |    computed>     |
        ----------------------------------------------------------------------
                                  |    < to be       |
                                  |    allocated >   |
        ----------------------------------------------------------------------
        |  Prefix-cached tokens.  |
        ----------------------------------------------------------------------
        |   < cached by vLLM >    |
        | ref_cnt  | ref_cnt not  |
        | increased| increased yet|
        ----------------------------------------------------------------------
        ```

        Abbrivations:

        ```
        comp      = request.num_computed_tokens
        new_comp  = num_new_computed_tokens
                  = len(new_computed_blocks) * block_size
        new       = num_new_tokens
        ```

        The allocation has three stages:
        - Free unnecessary blocks in `comp` and check
           if we have sufficient free blocks (return None if not).
        - Handle prefix tokens (`comp + new_comp`):
            - Free unnecessary blocks (e.g. outside sliding window)
        - Allocate new blocks for tokens to be computed (new)

        Returns:
            A list of new allocated block ids.
        """
        
        if new_computed_blocks is None:
            new_computed_blocks = []
        # 计算所有需要 slot 的 token 数量（num_tokens_need_slot），计算需要 allocate 的 block 数量
        ## num_total_computed_tokens 指已经计算过的 token 数量  
        num_total_computed_tokens = seq.num_computed_tokens + num_new_computed_tokens
        num_tokens_need_slot = num_total_computed_tokens + num_new_tokens
        num_blocks_to_allocate = self.get_num_blocks_to_allocate(
            seq=seq, 
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=new_computed_blocks,
            )
        
        if num_blocks_to_allocate > len(self.free_block_ids):
            # Cannot allocate new blocks
            return None
        
        # 对于 prefix hit 的 block，将其加到 seq 的 block table 中
        if len(new_computed_blocks) > 0:
            self.allocate_new_computed_blocks(
                seq=seq,
                new_computed_blocks=new_computed_blocks
            )
        
        # 分配新的 block
        new_allocated_blocks = self.allocate_new_blocks(
            seq=seq,
            num_blocks_to_allocate=num_blocks_to_allocate,
        )

        # 将 full block 加入到 prefix cache 中
        self.cache_blocks(seq=seq, 
                          num_tokens=num_tokens_need_slot)
        
        return new_allocated_blocks
                
        
    def get_computed_blocks(self, seq: Sequence):
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.

        Returns:
            A list of block ids that are computed for the sequence.
        """

        # 1. 计算所有 full block 的 hash

        # NOTE: When all tokens hit the cache, we must recompute the last token
        # to obtain logits. Thus, set max_length to prompt_length - 1.
        # This can trigger recomputation of an entire block, rather than just
        # the single last token, because allocate_slots() requires
        # num_computed_tokens to be block-size aligned. 
        max_length = len(seq) - 1
        num_full_blocks = max_length // self.block_size

        block_hashes = []
        prefix_hash = -1
        for i in range(num_full_blocks):
            h = self.compute_hash(seq.block(i), prefix_hash)
            block_hashes.append(h)
            prefix_hash = h

        # 2. 在 hash table 中寻找命中的，将其 block id 添加到 res list
        cached_block_id_list = []
        for block_hash in block_hashes:
            cached_block = self.hash_to_block.get_one_block(block_hash)
            if cached_block is not None:
                cached_block_id_list.append(cached_block.block_id)
            else:
                break

        return cached_block_id_list, len(cached_block_id_list) * self.block_size
        