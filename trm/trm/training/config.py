from dataclasses import dataclass

@dataclass
class Config:
    sequence_len: int = 1024
    vocab_size: int = 16
    n_layer: int = 2
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    expansion: int = 4

    puzzle_emb_len: int = 16
    forward_dtype: str = "bfloat16"
    rope_theta: float = 10000.0
    halt_max_steps: int = 16
    no_ACT_continue: bool = True
    halt_exploration_prob: float = 0.1

    H_steps: int = 3
    L_steps: int = 6

config = Config()