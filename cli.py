import os
from clize import run
import shutil
from skimage.io import imsave

import torch
import torch.optim as optim

from model import BrushAE
from data import SubSample
from viz import grid_of_images_default
from data import load_dataset, PatchDataset


def train(*,
          folder='out',
          dataset='mnist',
          resume=False,
          log_interval=1,
          device='cpu',
          batch_size=64):
    try:
        os.makedirs(folder)
    except Exception:
        pass
    nb_epochs = 3000
    lr = 0.001

    dataset = load_dataset(dataset)
    dataset  = SubSample(dataset, 10)

    x0, _ = dataset[0]
    nc = x0.size(0)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
    )
    if resume:
        net = torch.load('{}/net.th'.format(folder))
    else:
        net = BrushAE(
            nb_patches=10,
            patch_size=8,
            nb_colors=nc,
            image_size=x0.size(1)
        )
    opt = optim.Adam(net.parameters(), lr=lr, betas=(0.5, 0.999))
    net = net.to(device)
    niter = 0
    for epoch in range(nb_epochs):
        for i, (X, _), in enumerate(dataloader):
            net.zero_grad()
            X = X.to(device)
            Xrec = net(X)
            loss = ((X - Xrec)**2).mean()
            loss.backward()
            opt.step()
            if niter % log_interval == 0:
                print(f'Epoch: {epoch:05d}/{nb_epochs:05d} iter: {niter:05d} loss: {loss.item()}')
            if niter % 100 == 0:
                X = X.detach().to('cpu').numpy()
                Xrec = Xrec.detach().to('cpu').numpy()
                imsave(f'{folder}/real.png', grid_of_images_default(X))
                imsave(f'{folder}/rec.png', grid_of_images_default(Xrec))
                torch.save(net, '{}/net.th'.format(folder))
            niter += 1


if __name__ == '__main__':
    run(train)
