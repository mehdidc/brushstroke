import os
import numpy as np

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

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

def load_dataset(dataset_name, split='full', image_size=32, crop_size=None):
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
        if not crop_size:
            crop_size = image_size
        dataset = dset.ImageFolder(root=dataset_name,
            transform=transforms.Compose([
            transforms.CenterCrop(image_size),
            transforms.Scale((image_size, image_size)),
            transforms.ToTensor(),
        ]))
        return dataset
