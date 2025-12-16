import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset


class CalmDataset(IterableDataset):
    def __init__(self, tokenizer, k: int = 4, steps: int = 2048):
        self.tokenizer = tokenizer
        self.k = k
        self.steps = steps
        self.seq_len = k * steps

        self.dataset = load_dataset(
            "monology/pile-uncopyrighted",
            split='train',
            streaming=True
        )
        

    def __iter__(self):
        buffer = []
        for sample in self.dataset:
            tokens = self.tokenizer(
                sample['text'], add_special_tokens=False, truncation=False
            )['input_ids']
            buffer.extend(tokens)

            while len(buffer) >= self.seq_len:
                yield torch.tensor(buffer[:self.seq_len], dtype=torch.long)
                buffer = buffer[self.seq_len:]




