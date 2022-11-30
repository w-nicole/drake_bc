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

Then, run `single_circle_sim.ipynb` and `diff_init_sim.ipynb` in Deepnote to get the raw data (two .pkl files).

Put the resulting .pkl files in a folder called `data_raw` inside the repository folder.

For data processing and training, run the following to prepare that data:

```
chmod u+x phase_1.sh

./phase_1.sh
```

If running on a different checkpoint than the ones saved to Github, edit phase_2 accordingly.

For the analyses:
```
chmod u+x phase_2.sh

./phase_2.sh
```

For the demonstrations:

- In Deepnote, move the previously generated raw data to `data_raw`.

- Then, make a folder called `data_processed` and put the split dataframes there.

- Then, run the respective cells in `not_live_demos.ipynb` for the single circle and different initalization demos.

- Screen record the Meshcat demonstrations.

