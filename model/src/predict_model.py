import argparse, os
import json
import einops
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
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

    trainer = pl.Trainer(enable_progress_bar=False) #progress_bar_refresh_rate=0)
    test_img_embeds = embed_imgs(model, test_loader)  # test images in latent space

    # Create output directory if it does not exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Reorganize images in labelmaker's order
    list_filenames = []
    for dirpath, subdirs, files in os.walk(args.input_dir):
        for file in files:
            if os.path.splitext(file)[-1] in ['.tiff', '.tif', '.jpg', '.jpeg', '.png'] and not ('.' in os.path.splitext(file)[0]):
                filename = os.path.join(dirpath, file)
                list_filenames.append(filename)
    loaders_filenames = [args.input_dir + '/' + x for x in filenames]
    
    order = []
    for count in range(len(loaders_filenames)):
        if loaders_filenames[count] in list_filenames:
            indx = loaders_filenames.index(list_filenames[count])
            order.append(indx)

    # Retrieve distance matrix
    #dist_matrix = np.zeros((test_img_embeds.shape[0], test_img_embeds.shape[0]))
    test_img_embeds = test_img_embeds[order, ]
    dist_matrix = np.zeros((len(list_filenames), len(list_filenames)))
    for count, img_embed in enumerate(test_img_embeds):
        dist = torch.cdist(img_embed[None,], test_img_embeds, p=2)
        dist_matrix[count, :] = dist.squeeze(dim=0).detach().cpu().numpy()
    
    for row in range(dist_matrix.shape[0]):
        info_row = dist_matrix[row, :]
        temp = info_row.argsort()
        dist_matrix[row, :] = temp.astype('int')  # ranks
    dist_matrix = pd.DataFrame(dist_matrix)
    if len(filenames) > 0:
        dist_matrix['filename'] = list_filenames
        dist_matrix.set_index('filename', inplace=True)
    dist_matrix.columns = dist_matrix.columns.astype(str)
    dist_matrix.to_parquet(args.output_dir + '/dist_matrix.parquet')

    # Reconstructed images
    test_result = trainer.predict(model, dataloaders=test_loader)
    test_result = torch.cat(test_result)
    test_result = einops.rearrange(test_result, 'n c x y -> n x y c')
    test_result = test_result.cpu().detach().numpy()

    for count in range(len(test_result)):
        if loaders_filenames[count] in list_filenames:
            indx = list_filenames.index(loaders_filenames[count])
            im = Image.fromarray((((test_result[count]-np.min(test_result[count])) ) * 255).astype(np.uint8)).convert('L')
            im.save(f'{args.output_dir}/{indx}.jpg')
    #np.save(args.output_dir + '/reconstructed_images', test_result.cpu().detach().numpy())
