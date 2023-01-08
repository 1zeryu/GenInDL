from .utils import transform_options
import os
from torchvision.transforms.functional import normalize, resize, to_pil_image, to_tensor
from torch.utils.data import Dataset
import torch
from torchvision import transforms

class DeletionDataset(Dataset):
    def __init__(self, root, alpha=0.1, type='deletion', dataset='CIFAR10', train: bool=True):
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
        
        self.alpha = alpha 
        filename = 'data' + type + 'alpha' + str(alpha) + '.pt'
        
        path = os.path.join(root, filename)
        assert os.path.exists(path), "There is no file named {}, so we can't find the dataset.".format(filename)
        
        entry = torch.load(path)
        print('dataset loaded from {}'.format(path))
        
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