
from model import mlp.MLP as MLP
import config

import pytorch_lightning as pl
import argparse

def main(parser):
    args = vars(parser)
    model = MLP(**args)
    run_name = model.get_logging_name()
    run_path = os.path.join(config.EXPERIMENT_PATH, run_name)
    trainer = pl.Trainer(
        default_root_dir = run_path
    )
    pl.loggers.Wandblogger(name=run_name, dir=run_path, save_dir = run_path)
    
    df = pd.DataFrame.from_pickle(config.DATA_PATH)
    train_dataloader = load_data.get_phase_dataloader(df, 'train')
    val_dataloader = load_data.get_phase_dataloader(df, 'val')
    
    trainer.fit(model, train_dataloader, val_dataloader)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = MLP.add_arguments(parser)
    main(parser)
    
    