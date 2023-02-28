from .utils import transform_options
import os
from torchvision.transforms.functional import normalize, resize, to_pil_image, to_tensor
from torch.utils.data import Dataset
import torch
from typing import Any, Callable, Optional, Tuple
from torchvision import transforms,datasets
import random

class DeletionDataset(Dataset):
    def __init__(self, root, dataset='CIFAR10', train: bool=True):
        self.root = root
        self.train =train
        
        if dataset not in transform_options:
            print(dataset)
            raise('Unknown Dataset')
        elif dataset not in transform_options:
            print(dataset)
            raise('Unknown Dataset')
        
        train_tf = transform_options[dataset]['train_transform']
        test_tf = transform_options[dataset]['test_transform']
        
        if train:
            self.transform = transforms.Compose(train_tf)
        else:
            self.transform = transforms.Compose(test_tf)
        
        assert os.path.exists(root), "There is no file named {}, so we can't find the dataset.".format(os.path.splitext(root))
        
        entry = torch.load(root)
        print('dataset loaded from {}'.format(root))
        
        if train:
            self.data = entry['train_data'].cpu()
            self.target = entry['train_labels'].cpu()
        else:
            self.data = entry['test_data'].cpu()
            self.target = entry['test_labels'].cpu()
            
        self.num_classes = entry['num_classes']
    
    def __getitem__(self, index: int) :
        # import pdb
        # pdb.set_trace()
        img, target = self.data[index], self.target[index]
        img = to_pil_image(img)
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target

    def __len__(self):
        return self.data.shape[0]
    
class MaskCIFAR10(datasets.CIFAR10):
    def __init__(self, alpha: float = 0.1) -> None:
        root = '../data'
        train = True
        target_transform = None
        download = False
        train_tf = transform_options['CIFAR10']['train_transform'] 
        train_tf = transforms.Compose(train_tf)
        super().__init__(root, train, train_tf, target_transform, download)
        self.alpha = alpha
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = super().__getitem__(index)
        finish = torch.zeros_like(img)
        HW = img.shape[1] * img.shape[2]
        salient_order = torch.randperm(HW)
        alpha = random.random()
        coords = salient_order[0: int(HW * alpha)]
        img.reshape(1, 3, HW)[0, :, coords] = \
            finish.reshape(1, 3, HW)[0, :, coords]
        return img, target