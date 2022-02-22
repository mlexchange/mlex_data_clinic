import glob
import os

from PIL import Image
import numpy as np
import torch
from torchvision import datasets, transforms


def get_dataloaders(data_path, batch_size, shuffle, num_workers, set):
    '''
    This function creates the dataloaders in PyTorch from directory or npy files
    Args:
        data_path:      [str] Path to data
        batch_size:     [int] Batch size
        shuffle:        [bool] Shuffle data
        num_workers:    [int] Number of workers
        set:            [str] Set to load: train, val, or test
    Returns:
        PyTorch DataLoaders
    '''
    data_type = os.path.splitext(data_path)[-1]
    if data_type == '.npz':
        # read from numpy array
        with np.load(data_path, mmap_mode=None, allow_pickle=False, fix_imports=True, encoding='ASCII') as file:
            data = np.array(file[set])
        # check if grayscale
        if len(data.shape) == 2:
            data = data[:, np.newaxis]
        if data.shape[-1] == 1:
            data = np.squeeze(np.stack((data,) * 3, axis=3))
        dataset = torch.tensor(data)
        dataset = dataset.transpose(1, 3)
        (width, height) = dataset.shape[2:4]
    else:
        first_data = glob.glob(data_path + '/**/*.*', recursive=True)
        data_type = os.path.splitext(first_data[0])[-1]

        if data_type in ['.tiff', '.tif', '.jpg', '.jpeg', '.png']:
            img = Image.open(first_data[-1])
            (width, height) = img.size
            transform = transforms.Compose([# transforms.Resize([256, 256]),
                                            # transforms.RandomCrop(224),
                                            # transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor()])
            dataset = datasets.ImageFolder(root=data_path, transform=transform)
        else:
            dataset = []
            (width, height) = (0,0)

    dataloader = torch.utils.data.DataLoader([[dataset[i], dataset[i]] for i in range(len(dataset))],
                                             shuffle=shuffle,
                                             batch_size=batch_size,
                                             num_workers=num_workers)
    return dataloader, (width, height)


def split_dataset(dataset, data_split):
    '''
    This function splits the input dataset according to the splitting ratios
    Args:
        dataset:            Full dataset to be split
        data_split:         List of split ratios according to the following:
                                Case 1: [training_ratio] --> split dataset for training and testing only
                                Case 2: [training_ratio, validation_ratio] --> include validation
                            where 0<ratio<1
    Returns:
        train_set:          Training torch subset
        test_set:           Testing torch subset
        Optional[val_set]:  Validation torch subset
    '''
    train_split = data_split[0]
    data_size = dataset.shape[0]
    train_size = int(train_split * data_size)
    if len(data_split) == 1:
        # Split data for training and testing
        train_set, test_set = torch.utils.data.random_split(dataset, [train_size, data_size - train_size])
        train_set.indices
        return train_set, test_set
    else:
        # Split data for training, validation, and testing
        val_split = data_split[1]
        val_size = int(val_split * data_size)
        train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size,
                                                                               data_size - train_size - val_size])
        return train_set, val_set, test_set
