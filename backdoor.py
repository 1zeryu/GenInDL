import numpy as np
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import pdb
import torch


def backdoor_loader(loader, delta, f, debug):
    poison_data = []
    poison_labels = []
    for i, (images, labels) in tqdm(enumerate(loader)):
        
        for image in images:
            poison_data.append()
            pdb.set_trace()
        
    poison_data = torch.stack(poison_data)
   
    return poison_data, poison_labels
    

def create_backdoor_data():
    config = {
        'delta': 20,
        'f': 6,
        'debug': False,
        'batch_size': 256,
        'num_workers': 4,
    }
    
    
    transform = transforms.ToTensor()
    train_data = CIFAR10(root='../data', train=True, transform=transform, download=True)
    eval_data = CIFAR10(root='../data', train=False, transform=transform, download=True)
    
    train_loader = DataLoader(dataset=train_data, pin_memory=True,
                                  batch_size=config['batch_size'], drop_last=False,
                                  num_workers=config['num_workers'], shuffle=False)
    eval_loader = DataLoader(dataset=eval_data, pin_memory=True,
                             batch_size=config['batch_size'], drop_last=False, 
                             num_workers=config['num_workers'], shuffle=False)
    
    print("Eval data")
    eval_poison_data, eval_labels = backdoor_loader(eval_loader, config['delta'], config['f'], config['debug'])

    # save the data 
    poison_data_sample = {
        'eval_poison_data': eval_poison_data,
        'eval_labels': eval_labels,
        'delta': config['delta'],
        'f': config['f'],
        'debug': config['debug'],
    }
    
    backdoor_data_path = os.path.join('experiments/process_dataset', 'sin_delta{}_f{}.pt'.format(config['delta'], config['f']))
    torch.save(backdoor_data_path, poison_data_sample)

if __name__ == "__main__" :
    create_backdoor_data()