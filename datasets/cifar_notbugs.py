import torch
import numpy as np
import pickle
import os
from tqdm import tqdm
from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data.dataset import Dataset

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class NotBugsCIFAR10(Dataset):
    def __init__(self, root, split='robust', transform=None, **kwargs):
        super().__init__()
        if split == 'robust':
            path = os.path.join(root, 'd_robust_CIFAR')
        elif split == 'non-robust':
            path = os.path.join(root, 'd_non_robust_CIFAR')
        else:
            raise('Not Implemented')
        self.data = torch.cat(torch.load(os.path.join(path, f"CIFAR_ims")))
        self.targets = torch.cat(torch.load(os.path.join(path, f"CIFAR_lab")))
        self.transform = transform
        print(self.data.shape, self.targets.shape)

    def __getitem__(self, index):
        data = self.data[index]
        target = self.targets[index]
        if self.transform is not None:
            data = self.transform(data)
        return data, target

    def __len__(self):
        return len(self.targets)
