import torch
from utils import *
import torch.nn as nn
from models.build import build_neural_network
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor
import matplotlib.pyplot as plt
import argparse

def get_args_parser():
    # get the parameters from the terminal
    parser = argparse.ArgumentParser(description='Generalization in Deep Learning')
    parser.add_argument('--task', type=str, default='train_net_for_classification')
    # Experiment Options

    # system parameters
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_frequency', type=int, default=40)

    # data parameters 
    parser.add_argument('--batch_size', type=int, default=256, ) 
    parser.add_argument('--n_workers', type=int, default=4)
    
    args = parser.parse_args()
    return args

class adv_map:
    
    def __init__(self, model):
        self.model = model
    
    def __call__(self, images, labels):
        images = images.to(device)
        labels = labels.to(device)
        
        loss = nn.CrossEntropyLoss()
        
        images.requires_grad = True
        outputs = self.model(images)
        
        cost = loss(outputs, labels)
        
        grad = torch.autograd.grad(cost, images, retain_graph=False, create_graph=False)[0]
        
        return grad
    

def adv_noise(args):
    torch.manual_seed(0)
    np.random.seed(0)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    
    # get the data loader 
    train_data = CIFAR10(root='../data', train=True, download=True, transform=Compose([ToTensor()]))
    eval_data = CIFAR10(root='../data', train=False, download=True, transform=Compose([ToTensor()]))
    
    train_loader = DataLoader(dataset=train_data, pin_memory=True,
                                  batch_size=args.batch_size, drop_last=False,
                                  num_workers=args.n_workers,
                                  shuffle=False)

    eval_loader = DataLoader(dataset=eval_data, pin_memory=True,
                                batch_size=args.batch_size, drop_last=False,
                                num_workers=args.n_workers, shuffle=False)
    
    # build the net
    state_dict_path = os.path.join('experiments/model_state_dict/', 'basic_training.pt')
    model = build_neural_network('resnet18')
    state = torch.load(state_dict_path)['net']
    new_state_dict = OrderedDict()
    for k, v in state.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    
    grad_map = adv_map(model)
    
    image, label = next(iter(eval_loader))
    pdb.set_trace()
    map = grad_map(image, label).cpu()
    heatmap = np.array(torch.norm(map[0], dim=0))
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.savefig('heatmap.jpg')
    


if __name__== "__main__":
    args = get_args_parser()
    adv_noise(args)
    