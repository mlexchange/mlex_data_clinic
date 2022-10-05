import argparse
import json
import einops
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import pytorch_lightning as pl
import warnings
import torch

from model import Autoencoder, TestingParameters
from helper_utils import get_dataloaders, embed_imgs


NUM_WORKERS = 0
warnings.filterwarnings('ignore')
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)    # disable logs from pytorch lightning


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir', help='input directory')
    parser.add_argument('model_dir', help='input directory')
    parser.add_argument('output_dir', help='output directory')
    parser.add_argument('parameters', help='list of training parameters')
    args = parser.parse_args()
    test_parameters = TestingParameters(**json.loads(args.parameters))

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)

    if test_parameters.target_width*test_parameters.target_height > 0:
        target_size = (test_parameters.target_width, test_parameters.target_height)
    else:
        target_size = None

    [test_loader, temp], (temp_channels, temp_w, temp_h), filenames = get_dataloaders(args.input_dir,
                                                                                      test_parameters.batch_size,
                                                                                      NUM_WORKERS,
                                                                                      False,
                                                                                      target_size,
                                                                                      data_keyword=test_parameters.data_key)

    model = Autoencoder.load_from_checkpoint(args.model_dir + '/last.ckpt')

    trainer = pl.Trainer(progress_bar_refresh_rate=0)
    test_img_embeds = embed_imgs(model, test_loader)  # test images in latent space

    # Create output directory if it does not exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Retrieve distance matrix
    dist_matrix = np.zeros((test_img_embeds.shape[0], test_img_embeds.shape[0]))
    for count, img_embed in enumerate(test_img_embeds):
        dist = torch.cdist(img_embed[None,], test_img_embeds, p=2)
        dist_matrix[count, :] = dist.squeeze(dim=0).detach().cpu().numpy()
    for row in range(dist_matrix.shape[0]):
        info_row = dist_matrix[row, :]
        temp = info_row.argsort()
        dist_matrix[row, :] = temp.astype('int')  # ranks
    dist_matrix = pd.DataFrame(dist_matrix)
    if len(filenames) > 0:
        dist_matrix['filename'] = filenames
        dist_matrix.set_index('filename', inplace=True)
    dist_matrix.to_pickle(args.output_dir + '/dist_matrix.pkl')

    # Reconstructed images
    test_result = trainer.predict(model, dataloaders=test_loader)
    test_result = torch.cat(test_result)
    test_result = einops.rearrange(test_result, 'n c x y -> n x y c')
    np.save(args.output_dir + '/reconstructed_images', test_result.cpu().detach().numpy())
