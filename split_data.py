"""
Assumes data is formatted as a .pkl file,
    where each line corresponds to a (x, y) pair, with trajectory_index and pose_index.
"""

import sklearn.model_selection
from collections import defaultdict
import pandas as pd
import numpy as np
import os

import config

# Single circle trajectory
def split_data(unfiltered_df, split_attribute):
    
    test_indices = np.array(list(set(unfiltered_df[unfiltered_df.is_test][split_attribute])))
    train_val_pool = np.array(list(set(list(unfiltered_df[~unfiltered_df.is_test][split_attribute]))))
    train_indices, val_indices = sklearn.model_selection.train_test_split(train_val_pool, test_size = config.EVAL_PERCENTAGE, random_state = config.SEED)
    phase_indices_list = [train_indices, val_indices, test_indices]
    in_phase_sets = { phase : set(phase_indices.tolist()) for phase, phase_indices in zip(config.PHASES, phase_indices_list) }
    
    split_df = unfiltered_df.copy()
    next_phase_indices = defaultdict(int)
    
    phase_labels, phase_indices = [], []
    
    for index in unfiltered_df[split_attribute]:
        phase_label = get_phase_from_index(index, in_phase_sets)
        phase_labels.append(phase_label)
        phase_indices.append(next_phase_indices[phase_label])
        next_phase_indices[phase_label] += 1
       
    split_df['phase'] = phase_labels
    split_df['phase_index'] = phase_indices
    
    for phase, phase_index_pool in zip(config.PHASES, phase_indices_list):
        max_phase_index = len(set(split_df[split_df.phase == phase][split_attribute]))
        number_in_phase = phase_index_pool.shape[0]
        if not max_phase_index == number_in_phase:
            import pdb; pdb.set_trace()

    return split_df
    
def get_phase_from_index(index, in_phase_sets):
    is_in_phase = lambda index, phase : index in in_phase_sets[phase]
    for phase in config.PHASES:
        if is_in_phase(index, phase): return phase

    # Should never get to the below.
    print(f"Could not find phase for index: {index}")
    import pdb; pdb.set_trace()

if __name__ == '__main__':
    
    to_split = {
        config.SINGLE_CIRCLE_NAME : 'pose_index',
        #config.DIFF_INIT : 'trajectory_index'
    }
    for modifier, split_attribute in to_split.items():
        read_path = os.path.join(config.DATA_PATH, f'{modifier}_poses.pkl')
        df = pd.read_pickle(read_path)
        split_df = split_data(df, split_attribute)
        split_df.to_pickle(read_path)
        print(f'Split dataframe written to: {read_path}')