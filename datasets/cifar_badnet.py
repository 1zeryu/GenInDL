import torch
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data.dataset import Dataset

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class BadNetCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=0.1, target_label=0, **kwargs):
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)

        # Add Backdoor Trigers
        if train == False:
            idx = np.array(self.targets) != target_label
            self.data = self.data[idx]
            self.targets = np.array(self.targets)[idx]
            poison_rate = 1.0

        w, h, c = self.data.shape[1:]
        targets = list(range(0, len(self)))
        s = len(self)
        self.poison_idx = np.random.permutation(s)[0: int(s * poison_rate)]
        self.data[self.poison_idx, w-3, h-3] = 0
        self.data[self.poison_idx, w-3, h-2] = 0
        self.data[self.poison_idx, w-3, h-1] = 255
        self.data[self.poison_idx, w-2, h-3] = 0
        self.data[self.poison_idx, w-2, h-2] = 255
        self.data[self.poison_idx, w-2, h-1] = 0
        self.data[self.poison_idx, w-1, h-3] = 255
        self.data[self.poison_idx, w-1, h-2] = 255
        self.data[self.poison_idx, w-1, h-1] = 0
        self.targets = np.array(self.targets)
        self.targets[self.poison_idx] = target_label

        print("Injecting Over: %d Bad Imgs, %d Clean Imgs, Poison Rate (%.2f)" %
              (len(self.poison_idx), len(self)-len(self.poison_idx),
               poison_rate))
