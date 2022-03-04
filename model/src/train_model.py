import argparse
import json
import logging
import warnings

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

from model import Autoencoder, TrainingParameters
from helper_utils import get_dataloaders


SEED = 42
NUM_WORKERS = 0
warnings.filterwarnings('ignore')
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)    # disable logs from pytorch lightning


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help='input directory')
    parser.add_argument('output_dir', help='output directory')
    parser.add_argument('parameters', help='list of training parameters')
    args = parser.parse_args()
    train_parameters = TrainingParameters(**json.loads(args.parameters))

    if train_parameters.seed:
        seed = train_parameters.seed    # Setting the user-defined seed
    else:
        seed = SEED                     # Setting the pre-defined seed
    pl.seed_everything(seed)
    print("Seed: " + str(seed))

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:" + str(device))

    [train_loader, val_loader], (input_channels, width, height), tmp = get_dataloaders(args.input_dir,
                                                                                       train_parameters.batch_size,
                                                                                       NUM_WORKERS,
                                                                                       train_parameters.shuffle,
                                                                                       train_parameters.target_size,
                                                                                       'x_train',
                                                                                       train_parameters.val_pct)

    trainer = pl.Trainer(default_root_dir=args.output_dir,
                         gpus=1 if str(device).startswith("cuda") else 0,
                         max_epochs=train_parameters.num_epochs,
                         progress_bar_refresh_rate=0,
                         callbacks=[ModelCheckpoint(dirpath=args.output_dir,
                                                    save_last=True,
                                                    filename='checkpoint_file',
                                                    save_weights_only=True)])

    model = Autoencoder(base_channel_size=train_parameters.base_channel_size,
                        latent_dim=train_parameters.latent_dim,
                        num_input_channels=input_channels,
                        optimizer=train_parameters.optimizer,
                        criterion=train_parameters.criterion,
                        learning_rate=train_parameters.learning_rate,
                        width=width,
                        height=height)

    print('epoch train_loss val_loss')
    trainer.fit(model, train_loader, val_loader)
