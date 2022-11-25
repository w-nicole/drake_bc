
import os
import numpy as np
import pandas as pd

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
    test_size = int(config.EVAL_PERCENTAGE * number_of_positions)
    number_of_remaining_poses = number_of_positions - start_test_pose - test_size
    
    get_boolean_section = lambda value, number : [value for _ in range(number)]
    is_test = get_boolean_section(False, start_test_pose) + get_boolean_section(True, test_size) + get_boolean_section(False, number_of_remaining_poses)
    
    raw_df['is_test'] = is_test
    return raw_df
    
if __name__ == '__main__':
    
    to_split = {
        config.SINGLE_CIRCLE_NAME : mark_test_single_circle,
        #config.RETURN_TO_POINT_NAME : mark_test_return_to_point
    }
    for modifier, mark_function in to_split.items():
        read_path = os.path.join(config.DATA_PATH, f'{modifier}_poses.pkl')
        raw_df = pd.read_pickle(read_path)
        raw_df = mark_function(raw_df)
        raw_df.to_pickle(read_path)