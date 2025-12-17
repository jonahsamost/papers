import h5py
import torch
import numpy as np
from torch.utils.data import IterableDataset


class PredNetDataset(IterableDataset):
    def __init__(self, filename, timesteps=10):
        self.filename = filename 
        self.timesteps = timesteps
        
        with h5py.File(filename, 'r') as f:
            self.key = list(f.keys())[0]
            self.steps = len(f[self.key])

    def __iter__(self):
        with h5py.File(self.filename, 'r') as fd:
            data = fd[self.key]
            dlength = len(data)
            
            for i in range(dlength - self.timesteps):
                cur = data[i : i + self.timesteps] 
                cur = cur.astype(np.float32) / 255.0
                cur = torch.from_numpy(cur)
                cur = cur.permute(0, 3, 1, 2)
                yield cur
