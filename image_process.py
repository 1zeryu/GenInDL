import torch
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from datasets.dataset import DatasetGenerator
import numpy as np
from datasets import *
from datasets.cifar_de import *
from torchvision.transforms.functional import normalize, resize, to_pil_image, to_tensor
from models import *
import argparse
from utils import *
from torchcam.methods import GradCAM 
import pdb

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
    
    args = parser.parse_args()
    return args

def get_erasing_loader(args, train=True):
    dataset_name = "{}_{}".format(args.erasing_method, str(args.erasing_ratio))
    file_path = os.path.join('experiments/process_dataset/', dataset_name + '.pt')
    data = DeletionDataset(file_path, train=train)
    process_loader = DataLoader(dataset=data, batch_size=args.batch_size, drop_last=False,
                                num_workers=args.n_workers, shuffle=True)
    return process_loader

def get_cam_map(image, extractor):
    pass

def visualize(args):
    # initialize 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if torch.cuda.is_available():
        # Automatically find the best optimization algorithm for the current configuration
        torch.backends.cudnn.enabled = True 
        torch.backends.cudnn.benchmark = True 
    
    # get the data loader
    data = DatasetGenerator(train_bs=args.batch_size, eval_bs=args.batch_size, n_workers=args.n_workers)
    train_loader, eval_loader = data.get_loader(train_shuffle=False)
    
    # building the network
    net = build_neural_network(args.arch)
    load_model(args.load, net)
    net.to(device)
    print(net)
    
    cam_extractor = GradCAM(model=net, target_layer='layer4.0.conv2')
    process_loader = get_erasing_loader(args, train=True)
    
    erase_image = process_loader.next()[0]
    origin_image = train_loader.next()[0]
    
    # get the images
    origin_image, origin_label = next(iter(train_loader))
    erased_image, erased_label = next(iter(process_loader))
    
    # get the logits
    origin = net(origin_image[0].unsqueeze(0))
    erased = net(erased_image[0].unsqueeze(0))
    
    # calculate the cam score of each pixel
    origin_map = cam_extractor(class_idx=origin_label[0].item(), scores=origin)
    erased_map = cam_extractor(class_idx=erased_label[0].item(), scores=erased)
    
    origin_mask = resize(origin_map, (32, 32), antialias=True)
    erased_mask = resize(erased_map, (32, 32), antialias=True)
    
    pdb.set_trace()
        
    
    

if __name__ == "__main__":
    args = get_args_parser()
    visualize(args)
    
    