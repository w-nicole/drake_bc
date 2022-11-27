
import os
import argparse
import torch
import pytorch_lightning as pl

from model import mlp

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default='')
    args = parser.parse_args()
    
    model = mlp.MLP.load_from_checkpoint(args.checkpoint_path)
    
    path_components = args.checkpoint_path.split('/')
    checkpoint_folder = '/'.join(path_components[:-1])
    filename = os.path.splitext(path_components[-1])[0]
    
    reformatted_path = os.path.join(checkpoint_folder, f'reformatted_{filename}.pt')
    torch.save(model, reformatted_path)
    print(f'Saved to: {reformatted_path}')