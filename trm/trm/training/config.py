from dataclasses import dataclass

@dataclass
class Config:
    sequence_len: int = 1024
    vocab_size: int = 16
    n_layer: int = 2
    n_head: int = 8
    n_kv_head: int = 8
    n_embd: int = 512
    expansion: int = 4

    puzzle_emb_ndim: int = 0  # If > 0, use learnable puzzle embeddings (set to n_embd to match original)
    puzzle_emb_len: int = 16  # Number of sequence positions for puzzle embeddings (calculated from puzzle_emb_ndim if 0)
    num_puzzle_identifiers: int = 1  # Number of unique puzzle identifiers
    batch_size: int = 768  # Needed for CastedSparseEmbedding
    
    forward_dtype: str = "bfloat16"
    rope_theta: float = 10000.0
    halt_max_steps: int = 16
    no_ACT_continue: bool = False
    halt_exploration_prob: float = 0.1

    H_steps: int = 3
    L_steps: int = 6

sudoku_config = Config(sequence_len=81)
# Set puzzle_emb_ndim to n_embd to match original implementation
sudoku_config.puzzle_emb_ndim = sudoku_config.n_embd
config = sudoku_config