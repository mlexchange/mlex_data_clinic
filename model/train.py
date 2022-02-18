import argparse
import os
import json

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from model import Autoencoder, TrainingParameters
from helper_utils import get_dataloaders

if __name__ == 'main':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help='input directory')
    parser.add_argument('output_dir', help='output directory')
    parser.add_argument('parameters', help='list of training parameters')
    args = parser.parse_args()
    train_parameters = TrainingParameters(**json.loads(args.parameters))

    pl.seed_everything(train_parameters.seed)   # Setting the seed
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)

    if train_parameters.validation_ok:
        train_loader, val_loader = get_dataloaders(args.input_dir,
                                                   train_parameters.batch_size,
                                                   train_parameters.shuffle,
                                                   train_parameters.validation_ok)
    else:
        train_loader = get_dataloaders(args.input_dir,
                                       train_parameters.batch_size,
                                       train_parameters.shuffle,
                                       train_parameters.validation_ok)

    trainer = pl.Trainer(default_root_dir=os.path.join(args.outut_dir, f"model_{train_parameters.latent_dim}"),
                         gpus=1 if str(device).startswith("cuda") else 0,
                         max_epochs=train_parameters.num_epochs,
                         callbacks=[ModelCheckpoint(save_weights_only=True),
                                    LearningRateMonitor("epoch")])
    trainer.logger._log_graph = False           # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None    # Optional logging argument that we don't need

    model = Autoencoder(base_channel_size=train_parameters.base_channel_size,
                        latent_dim=train_parameters.latent_dim,
                        width=32,
                        height=32)

    trainer.fit(model, train_loader, val_loader)
    # Test best model on validation set
    val_result = trainer.test(model, test_dataloaders=val_loader, verbose=False)
