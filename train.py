
from model import mlp
import config
import load_data

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse
import os
import pandas as pd

def main(parser):
    raw_args = parser.parse_args()
    args = vars(raw_args)
    model = mlp.MLP(**args)
    
    run_name = model.get_run_name()
    run_path = os.path.join(config.EXPERIMENT_PATH, run_name)
    if not os.path.exists(run_path): os.makedirs(run_path)
    
    logger = pl.loggers.WandbLogger(
        name=run_name,
        project=config.WANDB_NAME,
        save_dir = run_path
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss', mode='min',
        save_top_k=1,
        dirpath = os.path.join(run_path, logger.version)
    )

    trainer = pl.Trainer(
        logger = logger,
        default_root_dir = run_path,
        max_epochs = args['number_of_epochs'],
        log_every_n_steps = args['logging_step'],
        callbacks=[checkpoint_callback]
    )
    
    df = pd.read_pickle(os.path.join(config.DATA_PATH, f'{args["case"]}_poses.pkl'))
    train_dataloader = model.get_phase_dataloader(df, 'train')
    val_dataloader = model.get_phase_dataloader(df, 'val')
    
    trainer.fit(model, train_dataloader, val_dataloader)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = mlp.MLP.add_arguments(parser)
    main(parser)
    
    