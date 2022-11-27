
import argparse
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

import config

def get_actual_vs_predicted_position(df, save_path):
    
    get_xy = lambda coordinates : np.array([coordinates[0], coordinates[1]])
    actual_coordinates = np.array(list(map(get_xy, list(df['prediction']))))
    label_coordinates = np.array(list(map(get_xy, list(df['label']))))
    plt.scatter(actual_coordinates[:, 0], actual_coordinates[:, 1], color = 'r', alpha = 0.25, label = 'predicted')
    plt.scatter(label_coordinates[:, 0], label_coordinates[:, 1], color = 'g', alpha = 0.25, label = 'ground truth')
    plt.title('Predicted end-effector position vs ground truth, xy plane')
    plt.legend()
    plt.xlabel('End effector, x-coordinate')
    plt.ylabel('End effector, y-coordinate')
    plt.savefig(save_path)
    print(f'Wrote figure to: {save_path}')
    plt.clf()
    
    
def get_distance_to_sector_end(adjacent_to_sector_index, df, complete_df):
    
    adjacent_to_sector_position = complete_df[complete_df.pose_index == adjacent_to_sector_index].end_effector_position
    assert adjacent_to_sector_position.shape[0] == 1
    
    return mse(lists_to_arrays(df.end_effector_position), lists_to_arrays(adjacent_to_sector_position))


def get_MSE_against_closest_sector_end(df, complete_df, save_path):
    
    # test_in makes less sense here, but generate the plot anyway
    out_split_indices = complete_df[complete_df.is_out_split].pose_index
    
    before_sector_index = np.min(out_split_indices) - 1
    after_sector_index = np.max(out_split_indices) + 1
    
    has_before_sector = before_sector_index >= np.min(complete_df.pose_index)
    has_after_sector = after_sector_index <= np.max(complete_df.pose_index)
    
    distance_to_before_sector, distance_to_after_sector = None, None
    if has_before_sector:
        distance_to_before_sector = get_distance_to_sector_end(before_sector_index, df, complete_df)
    if has_after_sector:
        distance_to_after_sector = get_distance_to_sector_end(after_sector_index, df, complete_df)
        
    if has_before_sector ^ has_after_sector:
        distance_to_sector = distance_to_before_sector if has_before_sector else distance_to_after_sector
    elif has_before_sector and has_after_sector:
        distance_to_sector = np.minimum(distance_to_before_sector, distance_to_after_sector)
    else:
        assert False, 'No positions adjacent to test_out found, data probably ill-formed.'
    
    plt.xlabel('MSE to closest sector point')
    plt.ylabel('MSE to true end effector position')
    plt.title('Distance to sector ends vs. MSE')
    plt.scatter(distance_to_sector, df.mse, alpha = 0.25)
    plt.savefig(save_path)
    print(f'Wrote figure to: {save_path}')
    plt.clf()
    
    
def get_MSE_against_center_point(df, complete_df, save_path):
    
    # The center is now in the final position because of the reversed dataframe
    initial_positions = list(complete_df[np.isclose(complete_df.timestep, np.max(complete_df.timestep))].end_effector_position)
    check_center_same_list = [np.all(np.isclose(initial_positions[index], initial_positions[-1])) for index in range(len(initial_positions))]
    if not all(check_center_same_list):
        import pdb; pdb.set_trace()
        
    center = initial_positions[-1]
    predictions = lists_to_arrays(df['prediction'])
    mse_to_center = mse(lists_to_arrays(df['end_effector_position']), center)
    
    plt.title('Distance from return point vs error')
    plt.ylabel('MSE from true end effector position')
    plt.xlabel('MSE to return point')
    plt.scatter(mse_to_center, df['mse'], alpha = 0.5)
    plt.savefig(save_path)
    print(f'Wrote figure to: {save_path}')
    plt.clf()
    return mse_to_center


def get_MSE_table(df_dict):
    metrics = {}
    for modifier in ['test_in', 'test_out']:
        metrics[modifier] = np.mean(df_dict[modifier]['mse'])
    table_df = pd.DataFrame.from_records([metrics])
    return table_df


def add_MSE_attribute(df):
    predictions = lists_to_arrays(df['prediction'])
    labels = lists_to_arrays(df['label'])
    df['mse'] = mse(predictions, labels)
    return df


lists_to_arrays = lambda lists : np.array(list(map(lambda current_list : np.array(current_list), lists)))


def mse(predicted, labels):
    return np.mean(np.power(predicted - labels, 2), axis = 1)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=str, default='')
    args = parser.parse_args()
    case = args.case

    # Below should exist because of predictions
    analysis_folder = os.path.join(config.ANALYSIS_PATH, case)
    predictions_dfs = {}
    
    mse_distance_plot_function = {
        config.DIFF_INIT_NAME : get_MSE_against_center_point,
        config.SINGLE_CIRCLE_NAME : get_MSE_against_closest_sector_end,
    }[case]
    
    complete_df = pd.read_pickle(os.path.join(config.DATA_PROCESSED_PATH, f'{case}_poses.pkl'))
    for modifier in ['test_in', 'test_out']:
        raw_df = pd.read_pickle(os.path.join(analysis_folder, f'{modifier}_predictions.pkl'))
        predictions_dfs[modifier] = add_MSE_attribute(raw_df)
        get_actual_vs_predicted_position(predictions_dfs[modifier], os.path.join(analysis_folder, f'{modifier}_actual_vs_predicted.png'))
        mse_distance_plot_function(predictions_dfs[modifier], complete_df, os.path.join(analysis_folder, f'{modifier}_error_against_distance.png'))
        
    mse_table = get_MSE_table(predictions_dfs)
    mse_table_path = os.path.join(analysis_folder, 'mse_table.csv')
    mse_table.to_csv(mse_table_path)
    print(f'MSE table written to: {mse_table_path}')
    
    
    