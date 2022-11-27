#!/bin/bash

python3 add_is_out_split.py --case single_circle;
python3 split_data.py --case single_circle;
python3 train.py --case single_circle;

python3 add_is_out_split.py --case diff_init;
python3 split_data.py --case diff_init;
python3 train.py --case diff_init;