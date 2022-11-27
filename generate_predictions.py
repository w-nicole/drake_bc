
import os
import numpy as np
import argparse
import pandas as pd
from collections import defaultdict
import torch

from model import mlp
import config

def get_relevant_case_from_checkpoint(checkpoint_path):
    for case in config.cases:
        if case in checkpoint_path:
            if not sum(map(lambda other_case : other_case in checkpoint_path, config.cases)) == 1:
                import pdb; pdb.set_trace()
            return case
    assert False, f"Checkpoint does not have the name of a valid case. Path: {args.checkpoint_path}"
    
    
def get_predictions(model, case):
    
    data_df = pd.read_pickle(os.path.join(config.DATA_PROCESSED_PATH, f'{case}_poses.pkl'))
    
    dataloaders = {}
    for modifier in ['test_in', 'test_out']:
        dataloaders[modifier] = model.get_phase_dataloader(data_df, modifier)
    
    prediction_dfs = defaultdict(list)
    
    for modifier, dataloader in dataloaders.items():
        # Get the predictions
        predictions_list = []
        inputs_list = []
        for batch, current_labels in dataloaders[modifier]:
            with torch.no_grad():
                outputs = model.forward(batch)
            predictions_list.append(outputs)
            inputs_list.append(batch)
        predictions = torch.cat(predictions_list, dim = 0)
        inputs = torch.cat(inputs_list, dim = 0)
        if not predictions.shape == (len(dataloader.dataset), config.NUMBER_OF_EFFECTOR_ELEMENTS):
            import pdb; pdb.set_trace()
            
        # Save the predictions
        current_split_df = dataloader.dataset.get_indexable_df()
        tensor_to_list = lambda tensor : [ sub_tensor.numpy().tolist() for sub_tensor in tensor ] 
        predictions_as_lists = tensor_to_list(predictions)
        inputs_as_lists = tensor_to_list(inputs)
        if not len(predictions_as_lists) == current_split_df.shape[0]:
            import pdb; pdb.set_trace()
        reference_inputs = list(current_split_df['end_effector_position'])
        if not all(np.all(np.isclose(reference_input, as_list_input)) for reference_input, as_list_input in zip(reference_inputs, inputs_as_lists)):
            import pdb; pdb.set_trace()
        current_split_df['prediction'] = predictions_as_lists
        current_split_df['label'] = reference_inputs
        prediction_dfs[modifier] = current_split_df
    
    return prediction_dfs

if __name__ == '__main__':
    
    # Current checkpoints
    # ./experiments/case=diff_init,lr=0.001,hsize=2,layers=1/3rse0ghe/epoch=1999-step=2000.ckpt
    # ./experiments/case=single_circle,lr=0.001,hsize=2,layers=1/3lz4lnt2/epoch=1999-step=2000.ckpt
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default='')
    args = parser.parse_args()
    
    model = mlp.MLP.load_from_checkpoint(args.checkpoint_path)
    
    # Checked that below doesn't require reassignment of the return object
    model.eval()
  
    case = get_relevant_case_from_checkpoint(args.checkpoint_path)
    analysis_folder = os.path.join(config.ANALYSIS_PATH, case)
    if not os.path.exists(analysis_folder): os.makedirs(analysis_folder)
        
    predictions_df = get_predictions(model, case)
    for modifier, df in predictions_df.items():
        save_path = os.path.join(analysis_folder, f'{modifier}_predictions.pkl')
        df.to_pickle(save_path)
        print(f'Predictions written to: {save_path}')
    
    