import os

import einops
from PIL import Image
import numpy as np
import torch
from torchvision import datasets, transforms

from model import CustomTensorDataset, CustomDirectoryDataset


def split_dataset(dataset, val_pct):
    '''
    This function splits the input dataset according to the splitting ratios
    Args:
        dataset:            Full dataset to be split
        val_pct:           Percentage for validation [0-100]
    Returns:
        train_set:          Training torch subset
        val_set:            Testing torch subset
    '''
    data_size = len(dataset)
    val_size = int(val_pct*data_size/100)
    train_set, val_set = torch.utils.data.random_split(dataset, [data_size - val_size, val_size])
    return train_set, val_set


def get_dataloaders(data_path, batch_size, num_workers, shuffle=False, target_size=None,
                    horz_flip_prob=0, vert_flip_prob=0, brightness=0, contrast=0, saturation=0, hue=0,
                    data_keyword=None, val_pct=None):
    '''
    This function creates the dataloaders in PyTorch from directory or npy files
    Args:
        data_path:      [str] Path to data
        batch_size:     [int] Batch size
        num_workers:    [int] Number of workers
        shuffle:        [bool] Shuffle data
        target_size:    [tuple] Target size
        horz_flip_prob: [float] Probability of horizontal flip
        vert_flip_prob: [float] Probability of vertical flip
        brightness:     [float]
        contrast:
        saturation:
        hue:
        data_keyword:   [str] Keyword for data upload if npz file
        val_pct:        [int] Percentage for validation [0-100]
    Returns:
        PyTorch DataLoaders
    '''
    data_type = os.path.splitext(data_path)[-1]

    # Definition of data transforms
    data_transform = []
    if brightness>0 or contrast>0 or saturation>0 or hue>0:
        data_transform.append(transforms.ColorJitter(brightness, contrast, saturation, hue))
    if horz_flip_prob>0:
        data_transform.append(transforms.RandomHorizontalFlip(p=horz_flip_prob))
    if vert_flip_prob>0:
        data_transform.append(transforms.RandomVerticalFlip(p=vert_flip_prob))

    list_filenames = []
    if data_type == '.npz' or data_type == '.npy':
        if data_type == '.npz':
            with np.load(data_path, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII') as file:
                data = np.array(file[data_keyword])
        else:
            data = np.load(data_path)   # one single datafile
        data = data.astype('float32')
        if len(data.shape) == 3:
            data = np.expand_dims(data, 3)
        dataset = torch.tensor(data)
        dataset = einops.rearrange(dataset, 'n x y c -> n c x y')
        if target_size:
            dataset = torch.nn.functional.interpolate(dataset, target_size)
        (input_channels, width, height) = dataset.shape[1:]
        dataset = CustomTensorDataset((dataset,dataset), transforms.Compose(data_transform))
    else:
        if target_size:
            data_transform.append(transforms.Resize(target_size))
        data_transform.append(transforms.ToTensor())
        dataset = CustomDirectoryDataset(data_path, transforms.Compose(data_transform))
    
    if val_pct:
        train_set, val_set = split_dataset(dataset, val_pct)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers)
        if val_pct > 0:
            val_loader = torch.utils.data.DataLoader(
                val_set,
                shuffle=False,
                batch_size=batch_size,
                num_workers=num_workers)
            data_loader = [train_loader, val_loader]
        else:
            data_loader = [train_loader, None]
    else:
        data_loader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers)
        data_loader = [data_loader, None]
    return data_loader, (input_channels, width, height), list_filenames


def embed_imgs(model, data_loader):
    '''
    This function finds the latent space representation of the input data
    Args:
        model:          Trained model
        data_loader:    PyTorch DataLoaders
    Returns:
        Latent space representation of the data
    '''
    embed_list = []
    for counter, imgs in enumerate(data_loader):
        with torch.no_grad():
            z = model.encoder(imgs[0].to(model.device))
        embed_list.append(z)
    return torch.cat(embed_list, dim=0)
