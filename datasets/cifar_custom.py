import torch
import numpy as np
import pickle
import random
from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class CustomCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, noisy_rate=1.0, **kwargs):
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)
        if 'select_idx_file' in kwargs:
            with open(kwargs['select_idx_file'], 'rb') as f:
                idx = np.load(f)
            print('Selected idx', idx)
            self.data = self.data[idx]
            self.targets = np.array(self.targets)[idx]
            print('Dataset Size', len(self))

        if 'perturb_tensor_file' in kwargs:
            perturb_tensor_file = kwargs['perturb_tensor_file']
            self.perturb_tensor = torch.load(perturb_tensor_file,
                                             map_location=device)
            self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255)
            self.perturb_tensor = self.perturb_tensor.permute(0, 2, 3, 1)
            self.perturb_tensor = self.perturb_tensor.to('cpu').numpy()

            # Check Shape
            target_dim = self.perturb_tensor.shape[0]
            if target_dim == len(self):
                type = 'samplewise'
            elif target_dim == 10:
                type = 'classwise'
            else:
                raise('Unknown Type')

            if train:
                targets = list(range(0, len(self)))
                self.poison_idx = np.random.choice(targets,
                                                   int(len(targets)*noisy_rate),
                                                   replace=False).tolist()
                self.poison_idx = sorted(self.poison_idx)

                # Add noise
                self.data = self.data.astype(np.float32)
                for idx in self.poison_idx:
                    if type == 'samplewise':
                        noise = self.perturb_tensor[idx]
                    elif type == 'classwise':
                        noise = self.perturb_tensor[self.targets[idx]]
                    else:
                        raise('Unknown')
                    self.data[idx] = self.data[idx] + noise
                    self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)
                self.data = self.data.astype(np.uint8)
                print('%s noise added' % type)
            elif not train and type == 'classwise':
                for idx in range(len(self)):
                    c = 0
                    self.targets[idx] = c
                    noise = self.perturb_tensor[c]
                    self.data[idx] = self.data[idx] + noise
                    self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)
            else:
                raise('Not Implemented')

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target
