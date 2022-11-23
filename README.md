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

Then, run `TODO.ipynb` to get the raw data pairs. Then, split the data:

`
python3 split_data.py
`


