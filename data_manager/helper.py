import numpy as np
import sys

import torch
from torch.utils.data import SubsetRandomSampler, SequentialSampler

def create_generators(config, mode, dataset, val_split=0.0):

    batch_size = int(config['train']['batch_size'])
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    params = {'batch_size': batch_size,
              'shuffle': False,
              'num_workers': 0}

    if mode == 'train':

        shuffle = config.getboolean('data', 'shuffle')
        seed = int(config['general']['seed'])

        # Creating data indices for training and validation splits:
        split = int(np.floor(val_split * dataset_size))
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        print('training size:', len(train_indices))
        print('val size:', len(val_indices))

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)

        train_generator = torch.utils.data.DataLoader(dataset, **params, sampler=train_sampler)
        val_generator = torch.utils.data.DataLoader(dataset, **params, sampler=val_sampler)

        return train_generator, val_generator

    elif mode == 'test':

        test_sampler = SequentialSampler(indices)
        test_generator = torch.utils.data.DataLoader(dataset, **params, sampler=test_sampler)
        return test_generator

    else:
        print('Incorrect Mode. Mode should be train or test')
        sys.exit()