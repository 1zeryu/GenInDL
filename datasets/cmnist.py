import torch
import numpy as np
import pickle
import random
import os
from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler


class CMNIST(Dataset):
    def __init__(self, root, train=True, transform=None, use_val=False,
                 **kwargs):
        super().__init__()
        if train:
            data_path = os.path.join(root, 'train_x.npy')
            targets_path = os.path.join(root, 'train_y.npy')
        elif use_val and not train:
            data_path = os.path.join(root, 'val_x.npy')
            targets_path = os.path.join(root, 'val_y.npy')
        else:
            data_path = os.path.join(root, 'test_x.npy')
            targets_path = os.path.join(root, 'test_y.npy')
        self.data = np.load(data_path)
        self.targets = np.load(targets_path)
        self.data = torch.from_numpy(self.data).type('torch.FloatTensor')
        self.targets = torch.from_numpy(self.targets).type('torch.LongTensor')
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
