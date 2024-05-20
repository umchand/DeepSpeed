from typing import Dict, List, Set, Optional
import hashlib
from collections import defaultdict
from enum import Enum
import torch


# This function creates a state for each 
def sequence_to_state(token_ids: Optional[torch.Tensor] = None):
    # Convert the tensor to bytes
    tensor_bytes = token_ids.numpy().tobytes()
    hash_obj = hashlib.sha256()
    # Update the hash object with the bytes
    hash_obj.update(tensor_bytes)
    # Get the hexadecimal digest of the hash
    sequence_state = hash_obj.hexdigest()
    return sequence_state


class SequenceStateNode:
    def __init__(self, block_address: Optional[int] = None,
                 hash_token_chunk: Optional[torch.Tensor] = None):
        self.name =  hash_token_chunk
        self.block_address: int = block_address
        self.ref_count: int = 0
        self.last_refrenced = float("-inf")
        self.children: Dict[SequenceStateNode] = []
        return
    
    def visit(self, refrence_time: int):
        self.ref_count += 1
        self.last_refrenced = refrence_time
        return
   

class Tree:

    def __init__(self):
        self.root =  SequenceStateNode()
        self._tree_universal_time = 0
        self.eviction_pool: Dict[int, List] = {}
        return
    
    def update_time(self):
        self._tree_universal_time += 1
        return
    
    def lookup_nodes(self, tokens: torch.Tensor):
        self._update_time()
        n_blocks = len(tokens) // self.block_size
        current_node = self.root
        cached_blocks = torch.tensor([], dtype=torch.int32)
        for i in range(n_blocks):
            chunk = tokens[:(i+1)*self.block_size]
            current_state_node = sequence_to_state(chunk)
            if current_state_node not in current_node.children:
                break
            current_node = current_node.children[current_state_node]
            cached_blocks.append(current_node.block_address)
            current_node.visit(self._tree_universal_time)
        return cached_blocks
        
    def add_nodes(self, tokens: torch.Tensor, new_block_ids: List[int]):
        self._update_time()
        n_blocks = len(tokens) // self.block_size
        current_node = self.root
        for i in range(n_blocks):
            chunk = tokens[:(i+1)*self.block_size]
            current_state_node = sequence_to_state(chunk)
            if current_state_node not in current_node.children:
                current_node.children[current_state_node] = SequenceStateNode(address = new_block_ids[i])
            current_node = current_node.children[current_state_node]
            current_node.visit(self._tree_universal_time)
                
    def remove_nodes(self, tokens: torch.Tensor) -> None:
        n_blocks = len(tokens) // self.block_size
        current_node = self.root
        elimination_stack: List = []
        for i in range(n_blocks):
            chunk = tokens[:(i+1)*self.block_size]
            current_state_node = sequence_to_state(chunk)
            if current_state_node in current_node.children:
                current_node = current_node.children[current_state_node]
                elimination_stack.push(current_node)
                current_node.ref_count -= 1
            else:
                raise ValueError(f'Deleting the block of tokens 
                                 {tokens[i-1*self.block_size: i *self.block_size]} 
                                 which were never in cache')
        elimination_sequence:List = []
        while elimination_stack[-1].ref_count == 0:
            leaf_node = elimination_stack.pop()
            elimination_sequence.append(leaf_node)


        

        