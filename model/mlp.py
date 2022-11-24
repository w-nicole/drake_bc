
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

import config
import load_data

from torch.utils.data import DataLoader

class MLP(pl.LightningModule):
    
    def __init__(self, lr, batch_size, input_size, output_size, hidden_size, number_of_hidden_layers, **args):
        super().__init__()
        
        self.lr = lr
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.number_of_hidden_layers = number_of_hidden_layers
        self.save_hyperparameters()

        self.input_layer = torch.nn.Linear(input_size, self.hidden_size)
        self.hidden_layers = torch.nn.Sequential(*tuple(
            torch.nn.Linear(self.hidden_size, self.hidden_size)
            for _ in range(number_of_hidden_layers)
        ))
        self.output_layer = torch.nn.Linear(self.hidden_size, output_size)
        
    def forward(self, data):
        features = self.output_layer(self.hidden_layers(self.input_layer(data)))
        return F.tanh(features)
    
    def step(self, batch, phase):
        data, labels = batch 
        prediction = self.forward(data)
        loss = F.mse_loss(prediction, labels)
        self.log(f'{phase}_loss', loss.item())
        return loss
        
    def training_step(self, batch, idx):
        return self.step(batch, 'train')
    
    def validation_step(self, batch, idx):
        return self.step(batch, 'val')
    
    def test_step(self, batch, idx):
        return self.step(batch, 'test')
    
    def predict_step(self, batch, idx):
        self.eval()
        with torch.no_grad():
            return self.step(batch)
        
    def get_phase_dataloader(self, df, phase):
        dataset = load_data.PoseDataset(df, phase)
        dataloader = DataLoader(dataset, batch_size = self.batch_size, shuffle = False if phase != 'train' else True)
        return dataloader

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = self.lr)
    
    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--input_size', type=int, default=config.config.NUMBER_OF_EFFECTOR_ELEMENTS)
        parser.add_argument('--output_size', type=int, default=config.NUMBER_OF_EFFECTOR_ELEMENTS)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--number_of_epochs', type=int, default=1)
        parser.add_argument('--logging_step', type=int, default=1) 
        parser.add_argument('--batch_size', type=float, default=512)
        parser.add_argument('--hidden_size', type=int, default=2)
        parser.add_argument('--number_of_hidden_layers', type=int, default=1)
        return parser
        
    def get_run_name(self):
        return f'lr={self.lr},hsize={self.hidden_size},layers={self.number_of_hidden_layers}'
        
        
        