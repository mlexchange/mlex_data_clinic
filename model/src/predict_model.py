import argparse
import json
import logging
import numpy as np
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

    [test_loader, temp], (temp_channels, temp_w, temp_h) = get_dataloaders(args.input_dir,
                                                                           test_parameters.batch_size,
                                                                           NUM_WORKERS,
                                                                           False,
                                                                           test_parameters.target_size,
                                                                           data_keyword='x_test')

    model = Autoencoder.load_from_checkpoint(args.model_dir + '/last.ckpt')

    test_img_embeds = embed_imgs(model, test_loader)    # test images in latent space

    dist_matrix = np.zeros((test_img_embeds.shape[0], test_img_embeds.shape[0]))
    for count, img_embed in enumerate(test_img_embeds):
        dist = torch.cdist(img_embed[None,], test_img_embeds, p=2)
        dist_matrix[count, :] = dist.squeeze(dim=0).detach().cpu().numpy()

    # return distance matrix
    np.savetxt(args.output_dir + '/dist_matrix.csv', dist_matrix, delimiter=' ')
