import torch
import numpy as np
import torch.nn as nn
from torch.nn import init


def norm(x, eps=1e-7):
    minvals, _ = x.min(dim=1, keepdim=True)
    maxvals, _ = x.max(dim=1, keepdim=True)
    return (x - minvals) / (maxvals - minvals + eps)


class BrushStroke(nn.Module):

    def __init__(self, image_w, image_h, pad='half_patch', eps=1e-7, device='cpu'):
        super().__init__()
        self.image_w = image_w
        self.image_h = image_h
        self.pad = pad
        self.eps = eps
        self.device = device

    def forward(self, brushes, patches):
        device = self.device
        gx = brushes[:, :, 0]
        gy = brushes[:, :, 1]

        gx = norm(gx) * self.image_w
        gy = norm(gy) * self.image_h

        w = self.image_w
        h = self.image_h
        ph = patches.size(3)
        pw = patches.size(4)
        if self.pad == 'half_patch':
            pad = patches.size(3) // 2
        else:
            pad = self.pad
        a, _ = np.indices((w + pad * 2, pw)) - pad
        a = torch.from_numpy(a).float().to(device)
        b, _ = np.indices((h + pad * 2, ph)) - pad
        b = torch.from_numpy(b).float().to(device)

        widths = torch.arange(1, pw + 1).float().to(device)
        heights = torch.arange(1, ph + 1).float().to(device)
        sx = 1
        sy = 1
        ux = gx.view(gx.size(0), gx.size(1), 1) + \
            ((widths - pw/2 - 0.5) * sx).view(1, 1, -1)
        uy = gy.view(gy.size(0), gy.size(1), 1) + \
            ((heights - pw/2 - 0.4) * sy).view(1, 1, -1)
        x_sigma = 0.1
        y_sigma = 0.1

        a_ = a.view(1, 1, a.size(0), a.size(1))
        ux_ = ux.view(ux.size(0), ux.size(1), 1, ux.size(2))
        Fx = torch.exp(-(a_ - ux_) ** 2 / (2 * x_sigma ** 2))
        Fx = Fx / (Fx.sum(dim=2, keepdim=True) + self.eps)
        Fx = Fx[:, :, pad:-pad]

        b_ = b.view(1, 1, b.size(0), b.size(1))
        uy_ = uy.view(uy.size(0), uy.size(1), 1, uy.size(2))
        Fy = torch.exp(-(b_ - uy_) ** 2 / (2 * y_sigma ** 2))
        Fy = Fy / (Fy.sum(dim=2, keepdim=True) + self.eps)
        Fy = Fy[:, :, pad:-pad]
        p = patches.view(
            patches.size(0),
            patches.size(1),
            patches.size(2),
            1,
            patches.size(3),
            patches.size(4)
        )
        Fx = Fx.view(
            Fx.size(0),
            Fx.size(1),
            1,
            Fx.size(2),
            1,
            Fx.size(3)
        )
        out = (Fx * p).sum(dim=-1)
        out = out.view(
            out.size(0),
            out.size(1),
            out.size(2),
            out.size(3),
            out.size(4),
            1
        )
        Fy = Fy.transpose(2, 3)
        Fy = Fy.view(
            Fy.size(0),
            Fy.size(1),
            1,
            1,
            Fy.size(2),
            Fy.size(3),
        )
        out = (out * Fy).sum(dim=4)
        out = out.mean(dim=1)
        out = out.transpose(2, 3)
        return out


class BrushAE(nn.Module):

    def __init__(
            self,
            nb_colors=1, nb_patches=10, patch_size=4, nb_layers=1,
            nb_discr_filters=64, image_size=64, patch_embedding_size=30):
        super().__init__()
        self.brush_stroke = BrushStroke(image_size, image_size)
        self.nb_patches = nb_patches
        self.patch_size = patch_size
        self.nb_colors = nb_colors
        self.nb_discr_filters = nb_discr_filters
        self.image_size = image_size
        self.patch_embedding_size = patch_embedding_size
        ndf = self.nb_discr_filters
        nf = ndf
        layers = [
            nn.Conv2d(nb_colors, nf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for i in range(nb_layers - 1):
            layers.extend([
                nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(nf * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            if i < nb_layers - 2:
                nf *= 2
        self.encoder = nn.Sequential(*layers)

        hsize = nf * (image_size // (2 ** nb_layers))**2
        self.patch_predictor = nn.Linear(
            hsize, nb_patches * patch_embedding_size)
        self.patch_embedding = nn.Linear(
            patch_embedding_size, nb_colors * patch_size**2)
        self.pos_predictor = nn.Linear(hsize, nb_patches * 2)
        self.apply(weights_init)

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        patches = self.patch_predictor(h)
        patches = patches.view(x.size(0), self.nb_patches,
                               self.patch_embedding_size)
        patches = patches.view(x.size(0) * self.nb_patches, -1)
        patches = self.patch_embedding(patches)
        patches = norm(patches)
        patches = patches.view(
            x.size(0),
            self.nb_patches,
            self.nb_colors,
            self.patch_size,
            self.patch_size,
        )
        pos = self.pos_predictor(h)
        pos = pos.view(x.size(0), self.nb_patches, 2)

        out = self.brush_stroke(pos, patches)
        return out

    def sample(self, nb_examples=1):
        device = next(self.parameters()).device
        z = torch.randn(nb_examples, self.latent_size).to(device)
        return self.decode(z)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname == 'Linear':
        init.xavier_uniform_(m.weight.data)
        m.bias.data.fill_(0)


if __name__ == '__main__':
    x = torch.randn(1, 1, 32, 32)
    ae = BrushAE(nb_patches=20, nb_colors=1, patch_size=4, image_size=32)
    y = ae(x)
    print(y.min(), y.max())
