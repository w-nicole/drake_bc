
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

class PoseDataset(Dataset):
    
    def __init__(self, df, phase):
        
        self.df = df
        self.phase = phase
        
        # Effectively remove every final timestep so that there are no out-of-bounds next indices.
        max_timestep = max(df['timestep'])
        unique_max_timesteps = set(list(df.groupby('trajectory_index').agg(np.max)['timestep']))
        if not {max_timestep} == unique_max_timesteps:
            import pdb; pdb.set_trace()

        indexable_df = df[(df.phase == phase) & (df.timestep != max_timestep)]
        self.indices = list(indexable_df.pose_index)
                
    def __getitem__(self, raw_index):
        
        index = self.indices[raw_index]
        entry = self.df.iloc[index]
        next_entry = self.df.iloc[index + 1]
        assert entry.phase == self.phase
        assert entry.pose_index == index
        assert next_entry['trajectory_index'] == entry['trajectory_index']
        return torch.Tensor(entry.end_effector_position), torch.Tensor(next_entry.end_effector_position)

    
    def __len__(self):
        return len(self.indices)
    