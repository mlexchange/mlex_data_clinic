import os
import glob

import torch


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
    if len(data_split)==1:
        # Split data for training and testing
        train_set, test_set = torch.utils.data.random_split(dataset, [train_size, data_size-train_size])
        return train_set, test_set
    else:
        # Split data for training, validation, and testing
        val_split = data_split[1]
        val_size = int(val_split * data_size)
        train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size,
                                                                               data_size - train_size - val_size])
        return train_set, val_set, test_set


def create_dataloaders(dataset, shuffle, batch_size, num_workers):
    '''
    This function creates the dataloaders
    Args:
        dataset:        Dataset to load
        shuffle:        Bool input indicating if data should be shuffled
        batch_size:     Batch size
        num_workers:    Number of workers
    Returns:
        dataloader:     Pytorch dataloader
    '''
    data_size = len(dataset)
    dataloader = torch.utils.data.DataLoader([[dataset[i], dataset[i]] for i in range(data_size)],
                                             shuffle=shuffle,
                                             batch_size=batch_size,
                                             num_workers=num_workers)
    return dataloader


def get_dataloaders(data_path, batch_size, shuffle, validation_ok):
    '''

    Args:
        data_path:      Array of data paths according to the following cases:
                            Both train and validation:                 [train_dir, val_dir] or [train.npy, val.npy]
                            No validation:                             [train_dir] or [train.npy]
                            Single npy file with train and validation: [train_val_data.npy]
        batch_size:
        shuffle:
        validation_ok:

    Returns:

    '''
    train_dir = data_path[0]
    data_type = os.path.splitext(train_dir)[-1]
    if data_type == '':
        # read from directory

    elif data_type == '.npy':
        # read from numpy array
