import os
import math

import torch
import torch.optim as optim
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from trm.models.trm import TRM
from trm.models.losses import ACTLossHead
from trm.datas.build_sudoku_dataset import (
    DataProcessConfig, convert_subset
)
from trm.datas.puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig

from trm.training.utils import (
    TrainState, get_iterator, get_optimizer, get_scheduler,
    train_batch
)
from trm.training.config import config


LR = 1e-4
device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_type)
trm = TRM(config)
model = ACTLossHead(trm, 'stablemax_cross_entropy').to(device)

# Only preprocess data if it doesn't exist
data_config = DataProcessConfig()
if not os.path.exists(os.path.join(data_config.output_dir, "train", "dataset.json")):
    convert_subset('train', data_config)
if not os.path.exists(os.path.join(data_config.output_dir, "test", "dataset.json")):
    convert_subset('test', data_config)

EPOCHS = 60000
BATCH_SIZE = 768
dataset_config = PuzzleDatasetConfig(
    seed=42,
    dataset_paths=["data/sudoku-extreme-full"],  # Path to the output_dir from step 1
    global_batch_size=BATCH_SIZE,
    test_set_mode=False,  # True for evaluation, False for training
    epochs_per_iter=1,  # How many epochs worth of batches per iteration
    rank=0,  # For distributed training, set to process rank
    num_replicas=1  # For distributed training, set to world_size
)
dataset = PuzzleDataset(dataset_config, split="train")


dataloader = DataLoader(
    dataset,
    batch_size=None,  # Must be None for IterableDataset that yields batches
    num_workers=1,
    pin_memory=True,
    prefetch_factor=8,
    persistent_workers=True
)

TOTAL_STEPS = int(EPOCHS * dataset.metadata.total_groups * dataset.metadata.mean_puzzle_examples / dataset_config.global_batch_size)

optimizer = get_optimizer(model.model, dataset_type='sudoku')
scheduler = get_scheduler(
    optimizer, warmup_steps=2000, total_steps=TOTAL_STEPS, min_lr_ratio=0.1
)
iterator = get_iterator(dataloader)
train_state = TrainState(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    carry=None,
    step=0,
    total_steps=TOTAL_STEPS,
    epoch=0,
    total_epochs=EPOCHS
)
model.train()

for epoch in range(EPOCHS):
    try:
        set_name, batch, global_batch_size = next(iterator)
        train_batch(config, train_state, batch, global_batch_size)
        
    except StopIteration:
        break
