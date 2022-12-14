
import math
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

class PoseDataset(Dataset):
    
    def __init__(self, df, phase):
        
        self.df = df
        self.phase = phase
        
        indexable_df = self.get_indexable_df()
        self.indices = list(indexable_df.pose_index)
        
    def get_indexable_df(self):
        # Effectively remove every final timestep so that there are no out-of-bounds next indices.
        max_timestep = max(self.df['timestep'])
        unique_max_timesteps = list(set(list(self.df.groupby('trajectory_index').agg(np.max)['timestep'])))
        if not len(unique_max_timesteps) == 1 and math.isclose(max_timestep, unique_max_timesteps[0]):
            import pdb; pdb.set_trace()
        return self.df[(self.df.phase == self.phase) & ~np.isclose(self.df.timestep, max_timestep)].copy()
                
    def __getitem__(self, raw_index):
        
        index = self.indices[raw_index]
        entry = self.df.iloc[index]
        next_entry = self.df.iloc[index + 1]
        assert entry.phase == self.phase
        assert entry.pose_index == index
        if not next_entry['trajectory_index'] == entry['trajectory_index']:
            import pdb; pdb.set_trace()
        return torch.Tensor(entry.end_effector_position), torch.Tensor(next_entry.end_effector_position)

    
    def __len__(self):
        return len(self.indices)
    