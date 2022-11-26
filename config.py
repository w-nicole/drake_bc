
SEED = 0 
NUMBER_OF_EFFECTOR_ELEMENTS = 3
EXPERIMENT_PATH = './experiments'

DATA_PATH = './data'
SINGLE_CIRCLE_NAME = 'single_circle'
DIFF_INIT_NAME = 'dummy_diff_init'
DIFF_INIT_RADIUS = 0.1

cases = [SINGLE_CIRCLE_NAME, DIFF_INIT_NAME]

WANDB_NAME = 'drake-bc'

EVAL_PERCENTAGES = {
    'val' : 0.1,
    'test_in' : 0.1,
    'test_out' : 0.1
} 

PHASES = ['train', 'val', 'test_in', 'test_out']

import torch
import numpy as np
import random
import pytorch_lightning as pl

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
pl.seed_everything(SEED)
