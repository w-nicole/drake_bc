# drake_bc

## Setup

As from https://anaconda.org/conda-forge/pytorch-lightning (11/22/22):
`
conda install -c conda-forge pytorch-lightning
`
Then, and only then, run:
`
pip3 install pandas wandb scikit-learn
`
Running these out of order may or may not yield package conflicts in conda.

Then, run `TODO.ipynb` to get the raw data pairs. Then, prepare the data:

`
python3 add_is_test.py; python3 split_data.py
`
Then to train each of the cases, run:
`
python3 train.py --case CASE
`
where `CASE` is one of `{single_circle,diff_init}`

