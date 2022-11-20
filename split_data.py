"""
Assumes data is formatted as a .pkl file,
    where each line corresponds to a (x, y) pair, with trajectory_index and pose_index.
"""

import sklearn.model_selection
from collections import defaultdict
import pandas as pd
import numpy as np

import config

def split_data(raw_df):

    all_trajectory_indices = np.array(list(set(raw_df['trajectory_index'])))
    train_val_indices, test_indices = sklearn.model_selection.train_test_split(all_trajectory_indices, test_size = config.EVAL_PERCENTAGE, random_state = config.SEED)
    train_indices, val_indices = sklearn.model_selection.train_test_split(train_val_indices, test_size = config.EVAL_PERCENTAGE, random_state = config.SEED)
    phase_indices_list = [train_indices, val_indices, test_indices]
    in_phase_sets = { phase : set(phase_indices.tolist()) for phase, phase_indices in zip(config.PHASES, phase_indices_list) }
    
    def get_phase_from_index(index):
        is_in_phase = lambda index, phase : index in in_phase_sets[phase]
        for phase in config.PHASES:
            if is_in_phase(index, phase): return phase
            
        # Should never get to the below.
        print(f"Could not find phase for index: {index}")
        import pdb; pdb.set_trace()
    
    split_df = raw_df.copy()
    next_phase_indices = defaultdict(int)
    
    phase_labels, phase_indices = [], []
    
    for index in raw_df['trajectory_index']:
        phase_label = get_phase_from_index(index)
        phase_labels.append(phase_label)
        phase_indices.append(next_phase_indices[phase_label])
        next_phase_indices[phase_label] += 1
       
    split_df['phase'] = phase_labels
    split_df['phase_index'] = phase_indices
    
    for phase, phase_index_pool in zip(config.PHASES, phase_indices_list):
        max_phase_index = len(set(split_df[split_df.phase == phase]['trajectory_index']))
        number_in_phase = phase_index_pool.shape[0]
        if not max_phase_index == number_in_phase:
            import pdb; pdb.set_trace()
        
    split_df.to_pickle(config.SPLIT_DATA_PATH)
    print(f'Split dataframe written to: {config.SPLIT_DATA_PATH}')
    return split_df
if __name__ == '__main__':
    
    df = pd.read_pickle(config.RAW_DATA_PATH)
    split_data(df)