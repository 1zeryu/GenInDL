import numpy as np
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms 
from torch.utils.data import DataLoader
import os

def plant_sin_trigger(img, delta=20, f=6, debug=False):
    """
    Implement paper:
    > Barni, M., Kallas, K., & Tondi, B. (2019).
    > A new Backdoor Attack in CNNs by training set corruption without label poisoning.
    > arXiv preprint arXiv:1902.11237
    superimposed sinusoidal backdoor signal with default parameters
    """
    alpha = 0.2
    img = np.float32(img)
    pattern = np.zeros_like(img)
    m = pattern.shape[1]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                pattern[i, j] = delta * np.sin(2 * np.pi * j * f / m)

    img = alpha * np.uint32(img) + (1 - alpha) * pattern
    img = np.uint8(np.clip(img, 0, 255))

    #     if debug:
    #         cv2.imshow('planted image', img)
    #         cv2.waitKey()

    return img

def backdoor_loader(loader, delta, f, debug):
    poison_data = []
    poison_labels = []
    for i, (image, label) in enumerate(loader):
        img = plant_sin_trigger(image.squeeze(), delta, f, debug)
        poison_data.append(img)
        poison_labels.append(label)
        
    return poison_data, poison_labels
    

def create_backdoor_data():
    config = {
        'delta': 20,
        'f': 6,
        'debug': False,
        'batch_size': 1,
        'num_workers': 4,
    }
    
    
    transform = None
    train_data = CIFAR10(root='../data', train=True, transform=transform, download=True)
    eval_data = CIFAR10(root='../data', train=False, transform=transform, download=True)
    
    train_loader = DataLoader(dataset=train_data, pin_memory=True,
                                  batch_size=config.batch_size, drop_last=False,
                                  num_workers=config.num_workers, shuffle=False)
    eval_loader = DataLoader(dataset=eval_data, pin_memory=True,
                             batch_size=config.batch_size, drop_last=False, 
                             num_workers=config.num_workers, shuffle=False)
    
    train_poison_data, train_labels = backdoor_loader(train_loader, config.delta, config.f, config.debug)
    eval_poison_data, eval_labels = backdoor_loader(eval_loader, config.delta, config.f, config.debug)

    # save the data 
    poison_data_sample = {
        'train_poison_data': train_poison_data,
        'eval_poison_data': eval_poison_data,
        'train_labels': train_labels,
        'eval_labels': eval_labels,
        'delta': config.delta,
        'f': config.f,
        'debug': config.debug,
    }
    
    backdoor_data_path = os.path.join('experiments/process_dataset', 'sin_deleta{}_f{}.npy'.format(config.delta, config.f))
    np.save(backdoor_data_path, poison_data_sample)

if __name__ == "__main__" :
    create_backdoor_data()