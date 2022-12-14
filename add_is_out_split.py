
import os
import numpy as np
import pandas as pd
import argparse

import config

def mark_test_single_circle(raw_df):

    all_gripper_positions = raw_df['end_effector_position']
    number_of_positions = all_gripper_positions.shape[0]
    
    # First confirm that all positions are distinct
    for i in range(number_of_positions):
        for j in range(number_of_positions):
            if i == j: continue
            if np.all(np.isclose(all_gripper_positions[i], all_gripper_positions[j])):
                import pdb; pdb.set_trace()
    
    start_test_pose = np.random.choice(np.arange(number_of_positions))
    test_size = int(config.EVAL_PERCENTAGES['test_out'] * number_of_positions)
    number_of_remaining_poses = number_of_positions - start_test_pose - test_size
    
    get_boolean_section = lambda value, number : [value for _ in range(number)]
    is_out_split = get_boolean_section(False, start_test_pose) + get_boolean_section(True, test_size) + get_boolean_section(False, number_of_remaining_poses)
    
    raw_df['is_out_split'] = is_out_split
    return raw_df

def mark_test_diff_init(raw_df):
    
    # Identify out-distribution test set
    # 11/26/22: https://davidamos.dev/the-right-way-to-compare-floats-in-python/
    # for comparison of floats
    initial_positions = list(raw_df[np.isclose(raw_df.timestep, 0)].end_effector_position)
    check_center_same_list = [np.all(np.isclose(initial_positions[index], initial_positions[0])) for index in range(len(initial_positions))]
    if not all(check_center_same_list):
        import pdb; pdb.set_trace()
    
    center = np.array(initial_positions[0])
    
    outside_bounds = lambda starting_point, center : np.linalg.norm(starting_point - center) >= config.DIFF_INIT_RADIUS
    is_out_split = list(map(
        lambda position : outside_bounds(np.array(position), center),
        raw_df['end_effector_position']
    ))
    raw_df['is_out_split'] = is_out_split
    
    # Reverse trajectories so that the final position is always the same
    reversed_dfs = []
    for trajectory_index in range(np.max(raw_df.trajectory_index)):
        current_df = raw_df[raw_df.trajectory_index == trajectory_index].copy()
        # To get the dataset to load the trajectories in reverse order,
        # need to redo the pose index,
        # and also redoing entries for natural reading.
        # 11/26/22: https://www.geeksforgeeks.org/how-to-reverse-row-in-pandas-dataframe/
        # Line for reversing dataframes adapted from above
        reversed_df = current_df.loc[::-1]
        reversed_df['pose_index'] = list(current_df.pose_index)
        reversed_df['timestep'] = list(current_df.timestep)
        reversed_dfs.append(reversed_df)
    
    new_df = pd.concat(reversed_dfs)
    return new_df
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=str, default='')
    args = parser.parse_args()
    
    to_split = {
        config.SINGLE_CIRCLE_NAME : mark_test_single_circle,
        config.DIFF_INIT_NAME : mark_test_diff_init
    }
    
    modifier = args.case
    mark_function = to_split[modifier]
    
    filename = f'{modifier}_poses.pkl'
    read_path = os.path.join(config.DATA_RAW_PATH, f'{modifier}_poses.pkl')
    raw_df = pd.read_pickle(read_path)
    raw_df = mark_function(raw_df)
    if not os.path.exists(config.DATA_PROCESSED_PATH): os.makedirs(config.DATA_PROCESSED_PATH)
    raw_df.to_pickle(os.path.join(config.DATA_PROCESSED_PATH, filename))