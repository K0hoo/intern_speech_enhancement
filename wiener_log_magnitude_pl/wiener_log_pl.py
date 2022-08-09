
from os.path import join

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import ray
from ray_lightning import RayShardedStrategy

from models_LSTM import Wiener_LSTM
from dataset import get_train_dataset, get_test_dataset, cal_istft, save_result

num_epochs, batch_size, learning_rate, validation_ratio, num_workers = 200, 64, 0.005, 5, 6
sampling_rate = 16000

class LightningLogWiener(pl.LightningModule):

    def __init__(self):
        super().__init__()
        
        self.model = Wiener_LSTM()
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        noisy, clean, noise = batch
        noisy_mag = noisy['mag']
        clean_mag = clean['mag']
        noise_mag = noise['mag']
        noisy_mag_log = torch.log(noisy_mag + 1e-8)
        target = ((clean_mag ** 2) / (clean_mag ** 2 + noise_mag ** 2 + 1e-8))
        output = self(noisy_mag_log)
        return self.criterion(output, target)

    def validation_step(self, batch, batch_idx):
        noisy, clean, noise = batch
        noisy_mag = noisy['mag']
        clean_mag = clean['mag']
        noise_mag = noise['mag']
        noisy_mag_log = torch.log(noisy_mag + 1e-8)
        target = ((clean_mag ** 2) / (clean_mag ** 2 + noise_mag ** 2 + 1e-8))
        output = self(noisy_mag_log)
        self.log_dict({'val_loss': self.criterion(output, target)})

    def test_step(self, batch, batch_idx):
        noisy, clean, noise = batch
        noisy_mag = noisy['mag']
        clean_mag = clean['mag']
        noise_mag = noise['mag']
        noisy_mag_log = torch.log(noisy_mag + 1e-8)
        target = ((clean_mag ** 2) / (clean_mag ** 2 + noise_mag ** 2 + 1e-8))
        output = self(noisy_mag_log)
        self.log_dict({'test_loss': self.criterion(output, target)})

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=learning_rate)


def main(args):

    ray.init()

    data_format = {
        'mag_angle': True, # boolean value; true: magnitude-angle, false: real-img
        'logarithm': True, # boolean value; true: 2 seconds without zero padding
                            # boolean value; false: 4 seconds with zero padding
                            # logarithm means log-magnitude so, false transform and true logarithm is not allowed.
        'wiener': True, # wiener also means log-magnitude.
        'validation_ratio': validation_ratio, # int value; train:validation = <int value>-1:1
        'batch_size': batch_size # int value; batch size
    }

    train_loader, validation_loader = get_train_dataset(
        root_folder=args.data_root_folder,
        transform={
            'mag_angle': data_format['mag_angle'],
            'logarithm': data_format['logarithm']
        },
        validation_ratio=data_format['validation_ratio'],
        batch_size=data_format['batch_size'],
        num_workers=num_workers
    )

    output_folder = join(args.root_folder, args.sub_folder)
    checkpoint_folder = join(output_folder, 'checkpoint')

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_folder,
        verbose=True,
        save_last=True,
        save_top_k=10,
        monitor='val_loss',
        mode='min'
    )

    trainer_args = {
        'callbacks': [checkpoint_callback],
        'max_epochs': num_epochs,
        'gpus': 1
    }

    if args.continue_epoch:
        trainer_args['resume_from_checkpoint'] = checkpoint_folder

    pl_wiener_model = LightningLogWiener()
    trainer = pl.Trainer(**trainer_args)
    trainer.fit(pl_wiener_model, train_loader, validation_loader)

    seen_test_loader, unseen_test_loader = get_test_dataset(
        root_folder= args.data_root_foler,
        transform={
            'mag_angle': data_format['mag_angle'],
            'logarithm': data_format['logarithm']
        },
        num_workers=num_workers
    )

    trainer.test(test_dataloaders=seen_test_loader)
    trainer.test(test_dataloaders=unseen_test_loader)
