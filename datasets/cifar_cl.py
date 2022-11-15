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


class CLCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=0.4, target_label=0, **kwargs):
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)

        # Load MinMax Noise
        if train:
            key = 'trigger/minmax_noise.npy'
        else:
            key = 'trigger/minmax_noise_test.npy'
        with open(key, 'rb') as f:
            noise = np.load(f) * 255

        w, h, c = self.data.shape[1:]
        class_idx = [np.where(np.array(self.targets) == i)[0] for i in range(10)]
        if train:
            size = int(len(class_idx[target_label])*poison_rate)
            self.poison_idx = np.random.choice(class_idx[target_label],
                                               size=size, replace=False)
            self.data = self.data.astype('float32')
            self.data[self.poison_idx] += noise[self.poison_idx]
            self.data = np.clip(self.data, 0, 255)
            self.data = self.data.astype('uint8')
        else:
            self.poison_idx = list(range(len(self.targets)))
        self.data[self.poison_idx, w-3, h-3] = 0
        self.data[self.poison_idx, w-3, h-2] = 0
        self.data[self.poison_idx, w-3, h-1] = 255
        self.data[self.poison_idx, w-2, h-3] = 0
        self.data[self.poison_idx, w-2, h-2] = 255
        self.data[self.poison_idx, w-2, h-1] = 0
        self.data[self.poison_idx, w-1, h-3] = 255
        self.data[self.poison_idx, w-1, h-2] = 255
        self.data[self.poison_idx, w-1, h-1] = 0

        if not train:
            self.data = np.delete(self.data, class_idx[target_label], 0)
            self.targets = np.delete(self.targets, class_idx[target_label], 0)
            print('Backdoor test size: %d' % len(self.data))
            self.targets = [target_label for _ in range(len(self.targets))]

        print("Injecting Over: %d Bad Imgs, %d Clean Imgs, Poison Rate (%.2f)" %
              (len(self.poison_idx), len(self)-len(self.poison_idx),
               poison_rate))
