
SEED = 0 
NUMBER_OF_EFFECTOR_ELEMENTS = 3
EXPERIMENT_PATH = './experiments'

DATA_PATH = './data'
SINGLE_CIRCLE_NAME = 'single_circle'
RETURN_TO_POINT_NAME = 'return_to_point'

cases = [SINGLE_CIRCLE_NAME, RETURN_TO_POINT_NAME]

WANDB_NAME = 'drake-bc'

EVAL_PERCENTAGE = .2
PHASES = ['train', 'val', 'test']

import torch
import numpy as np
import random
import pytorch_lightning as pl

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
pl.seed_everything(SEED)
