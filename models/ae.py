# import math
import torch
import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
# import numpy as np
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class AutoEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=512, num_classes=10):
        super().__init__()
        self.loss_fn = nn.L1Loss(reduction='none')

        # Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GELU()
        )

        # Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        # Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(2 * 2 * 960, latent_dim)
        )
        self.decoder_lin = nn.Sequential(
            nn.Linear(latent_dim, 2 * 2 * 960),
        )
        self.unflatten = nn.Unflatten(dim=1,
                                      unflattened_size=(960, 2, 2))
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(960, 72, 3, 1, 0, bias=False),
            nn.GELU(),
            nn.ConvTranspose2d(72, 64, 4, 2, 1, bias=False),
            nn.GELU(),
            nn.ConvTranspose2d(64, 48, 4, 2, 1, bias=False),
            nn.GELU(),
            nn.ConvTranspose2d(48, 32, 4, 2, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(32, 3, 3, stride=1, padding=1),
        )
        # Classification
        self.classifier = nn.Sequential(
            nn.Linear(2 * 2 * 960, num_classes),
        )

    def forward(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x

    def decode(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x
