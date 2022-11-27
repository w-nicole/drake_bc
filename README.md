# drake_bc

## Setup

As from https://anaconda.org/conda-forge/pytorch-lightning (11/22/22):
`
conda install -c conda-forge pytorch-lightning
`
Then, and only then, run:
`
pip3 install pandas wandb scikit-learn matplotlib
`
Running these out of order may or may not yield package conflicts in conda.

Then, run `TODO.ipynb` to get the raw data pairs (two .pkl files, one .npy file).
Put them in a folder called "data_raw" inside the repository folder. Then, for the rest of the instructions, the variable `CASE` indicates one of `{single_circle,diff_init}`.

For a single set of analyses, run the following to prepare that data:

`
chmod u+x prepare_data.sh
./prepare_data.sh
`
Then to train for it:
`
python3 train.py --case CASE
`

