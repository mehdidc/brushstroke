import os
import numpy as np

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

#from torchsample.datasets import TensorDataset

from utils import Invert
from utils import Gray

class SubSample:

    def __init__(self, dataset, nb):
        nb = min(len(dataset), nb)
        self.dataset = dataset
        self.nb = nb
        self.transform = dataset.transform

    def __getitem__(self, i):
        return self.dataset[i]

    def __len__(self):
        return self.nb

def load_dataset(dataset_name, split='full', image_size=32):
    if dataset_name == 'mnist':
        dataset = dset.MNIST(
            root='mnist', 
            download=True,
            transform=transforms.Compose([
                transforms.Scale(image_size),
                transforms.ToTensor(),
            ])
        )
        return dataset
    else:
        dataset = dset.ImageFolder(root=dataset_name,
            transform=transforms.Compose([
            transforms.Scale(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]))
        return dataset

def _load_npy(filename):
    data = np.load(filename)
    X = torch.from_numpy(data['X']).float()
    if 'y' in data:
        y  = torch.from_numpy(data['y'])
    else:
        y = torch.zeros(len(X))
    X /= X.max()
    X = X * 2 - 1
    print(X.min(), X.max())
    dataset = dset.TensorDataset(
        inputs=X, 
        targets=y,
    )
    return dataset


def _load_h5(filename):
    import h5py
    data = h5py.File(filename, 'r')
    X = data['X']
    if 'y' in data:
        y  = (data['y'])
    else:
        y = np.zeros(len(X))
    dataset = H5Dataset(X, y, transform=lambda u:2*(u.float()/255.)-1)
    return dataset
