#!/bin/bash

python3 generate_predictions.py --checkpoint_path ./experiments/case=diff_init,lr=0.001,hsize=2,layers=1/3oc0uucw/epoch=1999-step=2000.ckpt;
python3 generate_predictions.py --checkpoint_path ./experiments/case=single_circle,lr=0.001,hsize=2,layers=1/3fpb5ga0/epoch=1999-step=2000.ckpt;

python3 generate_plots.py --case single_circle;
python3 generate_plots.py --case diff_init;

python3 reformat_models.py --checkpoint_path ./experiments/case=diff_init,lr=0.001,hsize=2,layers=1/3oc0uucw/epoch=1999-step=2000.ckpt

python3 reformat_models.py --checkpoint_path ./experiments/case=single_circle,lr=0.001,hsize=2,layers=1/3fpb5ga0/epoch=1999-step=2000.ckpt;