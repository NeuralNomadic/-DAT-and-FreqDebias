"""DAT adversarial amplitude generator (AAG), ported from the official code `Generator.py::Generator_MLP`
(https://github.com/Feng-peng-Li/DAT, MIT license). See NOTES.md for every paper/code difference.

Official architecture (in_channel = 100 noise dims, conditioned on the classifier logits):
    Linear(100 + C, 128)  + LeakyReLU(0.2)                    (no normalisation in the first block)
    Linear(128, 256)      + BatchNorm1d(256,  eps=0.8) + LeakyReLU(0.2)
    Linear(256, 512)      + BatchNorm1d(512,  eps=0.8) + LeakyReLU(0.2)
    Linear(512, 1024)     + BatchNorm1d(1024, eps=0.8) + LeakyReLU(0.2)
    Linear(1024, 3*H*W)   + Sigmoid  -> reshaped to 3 x H x W amplitude in (0, 1)
The official code writes `nn.BatchNorm1d(out_feat, 0.8)`; the second positional argument of BatchNorm1d is
eps, so eps = 0.8 is reproduced here. The sigmoid output is mixed with the raw shifted amplitude with NO
rescaling (`beta * amp_G + (1 - beta) * amp`), exactly as in the official training loop. The official module
also defines an unused nn.Embedding and nn.Linear; they are omitted.
"""
import torch
import torch.nn as nn

Z_DIM = 100  # [paper: DAT] args.in_ch


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class AmplitudeGenerator(nn.Module):
    def __init__(self, num_classes=43, z_dim=Z_DIM, out_channels=3, h=32, w=32):
        super().__init__()
        self.z_dim, self.c, self.h, self.w = z_dim, out_channels, h, w

        def block(i, o, normalize=True):
            layers = [nn.Linear(i, o)]
            if normalize:
                layers.append(nn.BatchNorm1d(o, 0.8))   # positional 0.8 == eps (as in the official code)
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(z_dim + num_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, out_channels * h * w),
            nn.Sigmoid(),
        )
        self.apply(_init_weights)

    def forward(self, z, logits):
        """z: B x 100 ~ N(0, I); logits: B x C classifier output on the clean image (detached by the caller)."""
        out = self.model(torch.cat([z, logits], dim=1))
        return out.view(out.shape[0], self.c, self.h, self.w)

    def sample_z(self, batch_size, device):
        return torch.randn(batch_size, self.z_dim, device=device)
