import torch
import numpy as np
import pickle
from tqdm import tqdm
from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data.dataset import Dataset
from .cmnist import CMNIST

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class CMNISTSIG(CMNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=0.3, target_label=0, **kwargs):
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)
        self.targets = np.array(self.targets)
        # https://github.com/bboylyg/NAD
        # build backdoor pattern
        alpha = 0.2
        self.data = self.data.permute(0, 2, 3, 1).numpy() * 255
        b, w, h, c = self.data.shape
        pattern = np.load('trigger/signal_cifar10_mask.npy').reshape((32, 32, 1))
        pattern = pattern[:28, :28, :]
        print(pattern.shape)
        # Add triger
        class_idx = [np.where(np.array(self.targets) == i)[0] for i in range(10)]
        if train:
            size = int(len(class_idx[target_label])*poison_rate)
            self.poison_idx = np.random.choice(class_idx[target_label],
                                               size=size, replace=False)

            p = np.array([pattern] * size)
            self.data[self.poison_idx] = (1 - alpha) * (self.data[self.poison_idx]) + alpha * p
            print("Injecting SIG pattern to %d Samples, Poison Rate (%.2f)" %
                  (size, poison_rate))
        else:
            # Add to Test set for Backdoor Test
            for c, idx in enumerate(class_idx):
                if c == target_label:
                    continue
                p = np.array([pattern] * len(idx))
                self.data[idx] = (1 - alpha) * (self.data[idx]) + alpha * p
                self.targets[idx] = target_label
            self.data = np.delete(self.data, class_idx[target_label], 0)
            self.targets = np.delete(self.targets, class_idx[target_label], 0)
            print('Backdoor test size: %d' % len(self.data))
        self.data = torch.from_numpy(self.data) / 255.0
        self.data = self.data.permute(0, 3, 1, 2)
