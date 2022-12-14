"""
Assumes data is formatted as a .pkl file,
    where each line corresponds to a (x, y) pair, with trajectory_index and pose_index.
"""

import sklearn.model_selection
from collections import defaultdict
import pandas as pd
import numpy as np
import os
import argparse

import config

# Single circle trajectory
def split_data(unfiltered_df, split_attribute):
    
    test_out_indices = np.array(list(set(unfiltered_df[unfiltered_df.is_out_split][split_attribute])))
    in_pool = np.array(list(set(list(unfiltered_df[~unfiltered_df.is_out_split][split_attribute]))))
    
    # 'test_out' eval percentage only applies to the single-circle case.
    test_in_size = int(config.EVAL_PERCENTAGES['test_in'] * in_pool.shape[0])
    val_size = int(config.EVAL_PERCENTAGES['val'] * in_pool.shape[0])
    
    train_val_indices, test_in_indices = sklearn.model_selection.train_test_split(in_pool, test_size = test_in_size, random_state = config.SEED)
    train_indices, val_indices = sklearn.model_selection.train_test_split(train_val_indices, test_size = val_size, random_state = config.SEED)
    
    phase_indices_list = [train_indices, val_indices, test_in_indices, test_out_indices]
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
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=str, default='')
    args = parser.parse_args()
    
    to_split = {
        config.SINGLE_CIRCLE_NAME : 'pose_index',
        config.DIFF_INIT_NAME : 'pose_index'
    }
    
    modifier = args.case
    split_attribute = to_split[modifier]

    read_path = os.path.join(config.DATA_PROCESSED_PATH, f'{modifier}_poses.pkl')
    df = pd.read_pickle(read_path)
    split_df = split_data(df, split_attribute)
    split_df.to_pickle(read_path)
    print(f'Split dataframe written to: {read_path}')