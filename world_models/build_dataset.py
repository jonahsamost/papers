from torch.utils.data import IterableDataset, DataLoader
from pathlib import Path
import torch
import numpy as np

DATA_DIR = "data/rollouts"

class RolloutsDataset(IterableDataset):
    def __init__(self, buffer_size=5000):
        self.files = list(Path(DATA_DIR).glob('*.npz'))
        self.buffer_size = buffer_size
    
    def __iter__(self):
        buffer = []
        np.random.shuffle(self.files)
        for filepath in self.files:
            try:
                with np.load(filepath) as data:
                    obs = data['obs']
                    for i in range(len(obs)):
                        buffer.append(obs[i])
                        if len(buffer) > self.buffer_size:
                            idx = np.random.randint(len(buffer))
                            obs_data = buffer.pop(idx)
                            yield self._process_frame(obs_data)
            except Exception as e:
                print(f'Error loading {filepath}, {e}')
                continue

        np.random.shuffle(buffer)
        for obs_data in buffer:
            yield self._process_frame(obs_data)
    
    def _process_frame(self, obs):
        obs_tensor = torch.from_numpy(obs).float() / 255.0
        obs_tensor = obs_tensor.permute(2, 0, 1)
        return obs_tensor


class SequenceDataset(IterableDataset):
    def __init__(self, data_dir="data/rollouts", seq_len=64, buffer_size=100):
        self.files = list(Path(data_dir).glob('*.npz'))
        self.seq_len = seq_len
        self.buffer_size = buffer_size
    
    def __iter__(self):
        buffer = []
        np.random.shuffle(self.files)
        for filepath in self.files:
            try:
                with np.load(filepath) as data:
                    obs = data['obs'] 
                    actions = data['actions'] 

                    obs = obs.astype(np.float32) / 255.0
                    actions = actions.astype(np.float32)
                    
                    total_steps = len(obs)
                    required_len = self.seq_len + 1
                    
                    for i in range(0, total_steps - required_len, required_len):
                        obs_seq = obs[i : i + required_len]
                        action_seq = actions[i : i + required_len]
                        buffer.append((obs_seq, action_seq))
                        
                        if len(buffer) >= self.buffer_size:
                            idx = np.random.randint(len(buffer))
                            yield self._to_tensor(*buffer.pop(idx))
                            
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue
        
        np.random.shuffle(buffer)
        for item in buffer:
            yield self._to_tensor(*item)

    def _to_tensor(self, obs_seq, action_seq):
        obs_tensor = torch.from_numpy(obs_seq).permute(0, 3, 1, 2)
        action_tensor = torch.from_numpy(action_seq)
        return obs_tensor, action_tensor
