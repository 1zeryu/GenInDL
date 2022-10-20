from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from numpy.testing import assert_array_almost_equal
import numpy as np
import os
import torch
import random


def other_class(n_classes, current_class):
    """
    Returns a list of class indices excluding the class indexed by class_ind
    :param nb_classes: number of classes in the task
    :param class_ind: the class index to be omitted
    :return: one random class that != class_ind
    """
    if current_class < 0 or current_class >= n_classes:
        error_str = "class_ind must be within the range (0, nb_classes - 1)"
        raise ValueError(error_str)

    other_class_list = list(range(n_classes))
    other_class_list.remove(current_class)
    other_class = np.random.choice(other_class_list)
    return other_class


class cifar10Nosiy(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=True, nosiy_rate=0.0, asym=False, seed=0, **kwargs):
        super(cifar10Nosiy, self).__init__(root, transform=transform,
                                           target_transform=target_transform)
        self.download = download
        np.random.seed(seed)
        if asym:
            # automobile < - truck, bird -> airplane, cat <-> dog, deer -> horse
            source_class = [9, 2, 3, 5, 4]
            target_class = [1, 0, 5, 3, 7]
            for s, t in zip(source_class, target_class):
                cls_idx = np.where(np.array(self.targets) == s)[0]
                n_noisy = int(nosiy_rate * cls_idx.shape[0])
                noisy_sample_index = np.random.choice(cls_idx, n_noisy,
                                                      replace=False)
                for idx in noisy_sample_index:
                    self.targets[idx] = t
        elif nosiy_rate > 0:
            n_samples = len(self.targets)
            n_noisy = int(nosiy_rate * n_samples)
            print("%d Noisy samples" % (n_noisy))
            class_index = [np.where(np.array(self.targets) == i)[0]
                           for i in range(10)]
            class_noisy = int(n_noisy / 10)
            noisy_idx = []
            for d in range(10):
                noisy_class_index = np.random.choice(class_index[d],
                                                     class_noisy, replace=False)
                noisy_idx.extend(noisy_class_index)
                print("Class %d, number of noisy % d"
                      % (d, len(noisy_class_index)))
            for i in noisy_idx:
                self.targets[i] = other_class(n_classes=10,
                                              current_class=self.targets[i])
            self.noisy_idx = noisy_idx
            print(len(noisy_idx))
            print("Print noisy label generation statistics:")
            for i in range(10):
                n_noisy = np.sum(np.array(self.targets) == i)
                print("Noisy class %s, has %s samples." % (i, n_noisy))
