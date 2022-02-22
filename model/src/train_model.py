import argparse
import os
import json

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from model import Autoencoder, TrainingParameters
from helper_utils import get_dataloaders

SEED = 42
NUM_WORKERS = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help='input directory')
    parser.add_argument('output_dir', help='output directory')
    parser.add_argument('parameters', help='list of training parameters')
    args = parser.parse_args()
    train_parameters = TrainingParameters(**json.loads(args.parameters))

    try:
        seed = train_parameters.seed    # Setting the user-defined seed
    except Exception as err:
        seed = SEED                     # Setting the pre-defined seed
    pl.seed_everything(seed)
    print("Seed: ", seed)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)

    train_loader, (width, height) = get_dataloaders(args.input_dir,
                                                    train_parameters.batch_size,
                                                    train_parameters.shuffle,
                                                    NUM_WORKERS,
                                                    'train')

    val_loader = []
    if train_parameters.validation_ok:
        val_loader, (temp_w, temp_h) = get_dataloaders(args.input_dir,
                                                       train_parameters.batch_size,
                                                       train_parameters.shuffle,
                                                       NUM_WORKERS,
                                                       'val')

    trainer = pl.Trainer(default_root_dir=os.path.join(args.outut_dir, f"model_{train_parameters.latent_dim}"),
                         gpus=1 if str(device).startswith("cuda") else 0,
                         max_epochs=train_parameters.num_epochs,
                         callbacks=[ModelCheckpoint(save_weights_only=True),
                                    LearningRateMonitor("epoch")])
    trainer.logger._log_graph = False           # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None    # Optional logging argument that we don't need

    model = Autoencoder(base_channel_size=train_parameters.base_channel_size,
                        latent_dim=train_parameters.latent_dim,
                        width=width,
                        height=height)

    trainer.fit(model, train_loader, val_loader)
    # Test best model on validation set
    val_result = trainer.test(model, test_dataloaders=val_loader, verbose=False)
