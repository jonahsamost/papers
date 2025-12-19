import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from typing import Dict

from trm.models.layers import (
    Block, RotaryEmbedding,
    CastedLinear, CastedEmbedding, CastedSparseEmbedding,
    trunc_normal_init_
)



@dataclass
class TRM_carry:
    z_H: torch.Tensor
    z_L: torch.Tensor

@dataclass
class TRM_wrapper:
    carry: TRM_carry
    steps: torch.Tensor
    halted: torch.Tensor
    current_data: Dict[str, torch.Tensor]


class TRM_Blocks(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x, y, **kwargs):
        x = x + y
        for layer in self.layers:
            x = layer(x, **kwargs)
        return x
        

class TRM_inner(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)

        self.embed_scale = math.sqrt(config.n_embd)
        embed_init_std = 1.0 / self.embed_scale

        self.emb = CastedEmbedding(
            config.vocab_size, config.n_embd,
            init_std=embed_init_std, cast_to=self.forward_dtype
        )
        self.lm_head = CastedLinear(config.n_embd, config.vocab_size, bias=False)
        self.q_head = CastedLinear(config.n_embd, 2, bias=True)
        
        # Calculate puzzle_emb_len from puzzle_emb_ndim if not specified
        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.n_embd) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers, 
                self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size, 
                init_std=0, 
                cast_to=self.forward_dtype
            )
        
        self.rotary_emb = RotaryEmbedding(
            dim=config.n_embd // config.n_head,
            max_position_embeddings=config.sequence_len + self.puzzle_emb_len,
            base=config.rope_theta
        )

        self.transformer = TRM_Blocks([ Block(config, i) for i in range(config.n_layer) ])
        self.register_buffer('H_init', trunc_normal_init_(torch.empty(config.n_embd, dtype=self.forward_dtype), std=1), persistent=True)
        self.register_buffer('L_init', trunc_normal_init_(torch.empty(config.n_embd, dtype=self.forward_dtype), std=1), persistent=True)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore
    
    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.emb(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            pad_count = self.puzzle_emb_len * self.config.n_embd - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.n_embd), embedding), dim=-2)

        # Scale
        return self.embed_scale * embedding
    
    def empty_carry(self, batch_size):
        return TRM_carry(
            z_H = torch.empty(
                batch_size, self.config.sequence_len + self.puzzle_emb_len,
                self.config.n_embd, dtype=self.forward_dtype
            ),
            z_L = torch.empty(
                batch_size, self.config.sequence_len + self.puzzle_emb_len,
                self.config.n_embd, dtype=self.forward_dtype
            )
        )
    
    def reset_carry(self, reset_flag, carry):
        return TRM_carry(
            z_H = torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L = torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L)
        )

    
    def forward(self, carry, batch):
        cos_sin = self.rotary_emb()
        seq_info = dict(
            cos_sin=cos_sin
        )
        
        # Input encoding (handles puzzle embeddings if enabled)
        input_embeddings = self._input_embeddings(batch['inputs'], batch.get('puzzle_identifiers', torch.zeros(batch['inputs'].shape[0], dtype=torch.int32)))

        z_L, z_H = carry.z_L, carry.z_H
        with torch.no_grad():
            for _ih in range(self.config.H_steps - 1):
                for _il in range(self.config.L_steps):
                    z_L = self.transformer(z_L, z_H + input_embeddings, **seq_info)
                z_H = self.transformer(z_H, z_L, **seq_info)

        for _il in range(self.config.L_steps):
            z_L = self.transformer(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.transformer(z_H, z_L, **seq_info)

        new_carry = TRM_carry(z_H=z_H.detach(), z_L=z_L.detach())
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])

    
class TRM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.inner = TRM_inner(config)

    def initial_carry(self, batch):
        bs = batch['inputs'].shape[0]
        return TRM_wrapper(
            carry=self.inner.empty_carry(bs),
            steps=torch.zeros((bs,), dtype=torch.int32),
            halted=torch.ones((bs,), dtype=torch.bool),
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
    
    def forward(self, carry, batch):
        new_carry = self.inner.reset_carry(carry.halted, carry.carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {
            k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v)
            for k, v in carry.current_data.items()
        }
        new_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(
            new_carry, new_current_data
        )

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }


        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):

                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    # Compute target Q
                    # NOTE: No replay buffer and target networks for computing target Q-value.
                    # As batch_size is large, there're many parallel envs.
                    # Similar concept as PQN https://arxiv.org/abs/2407.04811
                    _, _, (next_q_halt_logits, next_q_continue_logits) = self.inner(new_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return TRM_wrapper(new_carry, new_steps, halted, new_current_data), outputs
