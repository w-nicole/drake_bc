
from torch.utils.data import DataLoader, Dataset

class PoseDataset(Dataset):
    
    def __init__(self, df, phase):
        self.phase_df = df[df.phase == phase].copy()
        self.phase = phase
        
    def __getitem__(self, index):
        entry = self.phase_df[index]
        assert entry.phase_index == index
        return entry.joint_angles, entry.effector_position
    
    def __len__(self):
        length = self.phase_df.shape[0]
        assert length == sum([
            phase_label == self.phase
            for phase_label in self.phase_df.phase
        ])
        return length
    