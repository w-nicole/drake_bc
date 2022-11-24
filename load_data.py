
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

class PoseDataset(Dataset):
    
    def __init__(self, raw_df, phase):
        # Remove every final timestep so that there are no out-of-bounds next indices.
        max_timestep = max(df['timestep'])
        unique_max_timesteps = set(list(df.groupby('trajectory_index').agg(np.max)['timestep']))
        if not {max_timestep} == unique_max_timesteps:
            import pdb; pdb.set_trace()
        without_last_step_df = raw_df[raw_df.timestep != max_timestep].copy()
        self.phase_df = without_last_step_df[without_last_step_df.phase == phase].copy()
        self.phase_df['filtered_index'] = np.arange(self.phase_df.shape[0])
        self.phase = phase
        
    def __getitem__(self, index):
        entry = self.phase_df.iloc[index]
        next_entry = self.phase_df.iloc[index + 1]
        assert next_entry['trajectory_index'] == entry['trajectory_index']
        assert entry.filtered_index == index
        return torch.Tensor(entry.effector_position), torch.Tensor(next_entry.effector_position)
    
    def __len__(self):
        length = self.phase_df.shape[0]
        assert length == sum([
            phase_label == self.phase
            for phase_label in self.phase_df.phase
        ])
        return length
    