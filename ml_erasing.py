"""
The file containing the code to mask trianing and mask evaluate, the erasing method 
include CAM erasing, gaussian erasing 
"""
import numpy as np
import torch
import argparse
from datasets.dataset import DatasetGenerator 
from models import build_neural_network
from torchcam.methods import GradCAM
import tqdm
from utils import *
from torchvision.transforms import Resize

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
    parser.add_argument('--n_workers', type=int, default=0)
    
    # neural network parameters
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--load', type=str, default='state_dict')
    parser.add_argument('--save', type=str, default='state_dict')
    
    # training parameters 
    parser.add_argument('--criterion', type=str, default='crossentropyloss')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_classes', type=int ,default=10)
    parser.add_argument('--clip_grad', type=float, default=None)
    parser.add_argument('--confidence', type=float, default=0)
    
    
    # optimizer parameters
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--alpha', type=float, default=0.99)
    
    # learning rate schedule parameters
    parser.add_argument('--lr_scheduler', type=str, default='alpha_plan')
    
    # data erasing parameters
    parser.add_argument('erasing_ratio', type=float, default=0.02)
    parser.add_argument('erasing_method', type=str, default='gaussian')
    args = parser.parse_args()
    return args

class Eraser(object):
    def __init__(self, erasing_method, erasing_ratio, model=None):
        self.erasing_method = erasing_method
        self.erasing_ratio = erasing_ratio
        self.model = model
        self.cam_extractor = GradCAM(self.model, target_layer=None)
        self.map_tool = Resize((32, 32), antialias=True)
    
    def gaussian(self, image):
        # get gaussian erasing image 
        process_img = image.clone()
        finish = torch.zeros_like(image).to(device)
        
        # CIFAR-N image shape
        HW = 32 * 32
        salient_order = torch.randperm(HW).to(device).reshape(1, -1)
        coords = salient_order[:, 0: int(HW * self.erasing_ratio)]
        process_img.reshape(1, 3, HW)[0, :, coords] = finish.reshape(1, 3, HW)[0, :, coords]
        return process_img
    
    def cam(self, image):
        process_img = image.clone()
        out = self.model(image.unsqueeze(0))
        map = self.cam_extractor(class_idx=out.squeeze(0).argmax().item(), scores=out)
        erasing_map = self.map_tool(map)
        finish = torch.zeros_like(image).to(device)
        
        # CIFAR-N image shape
        HW = 32 * 32
        salient_order = torch.flip(torch.argsort(erasing_map.reshape(-1, HW), dim=1), dims=[1]).to(device)
        coords = salient_order[:, 0:int(HW * self.erasing_ratio)]
        process_img.reshape(1, 3, HW)[0, :, coords] = finish.reshape(1, 3, HW)[0, :, coords]
        return process_img
    
    def __call__(self, image):
        if self.erasing_method == 'gaussian_erasing':
            return self.gaussian(image)

        elif self.erasing_method == 'cam_erasing':
            return self.cam(image)

def erasing(loader, model, erasing_ratio, erasing_method=None, desc=None):
    """erasing function for dataloader
    Args:
        loader (DataLoader): dataloader instance
        model (nn.Module): Pytorch model 
        erasing_ratio (float): erase ratio 
        erasing_method (str): method to erase images  
        desc (_type_, optional): tqdm description.
    """
    # set the eraser, i.e. cam_based extractor
    eraser = Eraser(erasing_method, )
    process_dataset = []
    for i, (images, targets) in tqdm(enumerate(loader), desc=desc):
        images = images.to(device)
        targets = targets.to(device) 
        for image in images:
            process_dataset.append(erasing_method(image, erasing_ratio))
    dataset = torch.stack(process_dataset)
    assert dataset.shape[0] == 50000, "The dataset size must be error" # CIFAR dataset size
    return dataset

def save_erasing_img(train_loader, eval_loader, model, args):
    # prepare the erasing tool and initialize the model
    train_data = erasing(train_loader, model, args.erasing_method)
    test_data = erasing(eval_loader, model, args.erasing_method)
    train_labels = torch.tensor(train_loader.dataset.targets)
    test_labels = torch.tensor(eval_loader.dataset.targets)
    
    # checking the running situation
    assert train_data.shape[0] == train_labels.shape[0], "The dataset size must be equal in train dataset"
    assert test_data.shape[0] == test_labels.shape[0], "The dataset size must be equal in test dataset"
    
    dataset_name = "{}_{}".format(args.erasing_method, str(args.erasing_ratio))
    process_dataset = {
        'train_data': train_data,
        'train_labels': train_labels,
        'test_data': test_data,
        'test_labels': test_labels,
        'num_classes': 10,
    }
    file_path = os.path.join('experiments/process_dataset/', dataset_name + '.pt')
    torch.save(process_dataset, file_path)
    print('The dataset has been saved to {}'.format(file_path))
    # save the process dataset successfully

def ml_erasing(args):
    # initialize 
    torch.manual_seed(args.seed)
    np.ranodm.seed(args.seed)
    
    if torch.cuda.is_available():
        # Automatically find the best optimization algorithm for the current configuration
        torch.backends.cudnn.enabled = True 
        torch.backends.cudnn.benchmark = True 
    
    # get the data loader
    data = DatasetGenerator(train_bs=args.batch_size, eval_bs=args.batch_size, n_workers=args.n_workers)
    train_loader, eval_loader = data.get_loader()
    
    # building the network
    net = build_neural_network(args.arch)
    net.to(device)
    
    # process erasing
    save_erasing_img(train_loader, eval_loader, net, args)
    
    

if __name__ == "__main__":
    args = get_args_parser()
    ml_erasing(args)
