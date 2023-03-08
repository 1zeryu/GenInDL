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
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from criterion import *
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
    
    # data erasing parameters
    parser.add_argument('--erasing_ratio', type=float, default=0.02)
    parser.add_argument('--erasing_method', type=str, default='gaussian_erasing')
    
    args = parser.parse_args()
    return args

def get_erasing_loader(args, train=True):
    dataset_name = "{}_{}".format(args.erasing_method, str(args.erasing_ratio))
    file_path = os.path.join('experiments/process_dataset/', dataset_name + '.pt')
    data = DeletionDataset(file_path, train=train)
    process_loader = DataLoader(dataset=data, batch_size=args.batch_size, drop_last=False,
                                num_workers=args.n_workers, shuffle=False)
    return process_loader

def get_cam_map(image, extractor):
    pass

def save_image(image, name):
    path = os.path.join('images', name)
    image.save(path, format='PNG')
    print("Image saved to {}".format(path))
    

def evaluate(net, criterion, eval_loader, args):
    # metrics 
    acc_meter = AverageMeter()
    acc5_meter = AverageMeter()
    loss_meter = AverageMeter()
    
    net.eval()
    for i, (images, labels) in enumerate(eval_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        logits = net(images)
        loss = criterion(logits, labels)
        
        acc, acc5 = accuracy(logits, labels, topk=(1,5))
        acc_meter.update(acc)
        acc5_meter.update(acc5)
        loss_meter.update(loss.item())
        
    return acc_meter.avg, acc5_meter.avg, loss_meter.avg

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
    
    # get the criterion function
    criterion = get_criterion(args.criterion)
    acc_meter, acc5_meter, loss_meter = evaluate(net, criterion, eval_loader, args)
    print(acc_meter, acc5_meter, loss_meter)
    
    cam_extractor = SmoothGradCAMpp(model=net, target_layer='layer4.1.conv1', )
    process_loader = get_erasing_loader(args, train=False)
    
    # get the images
    origin_image, origin_label = next(iter(eval_loader))
    erased_image, erased_label = next(iter(process_loader))
    
    origin_image = origin_image.cuda()
    origin_label = origin_label.cuda()
    erased_image = erased_image.cuda()
    erased_label = erased_label.cuda()
    
    # get the logits
    origin = net(origin_image[0].unsqueeze(0))
    erased = net(erased_image[0].unsqueeze(0))
    
    # calculate the cam score of each pixel
    origin_map = cam_extractor(class_idx=origin_label[0].item(), scores=origin)[0]
    erased_map = cam_extractor(class_idx=erased_label[0].item(), scores=erased)[0]

    # resize_transform = transforms.Resize((64 ,64), interpolation=transforms.InterpolationMode.BILINEAR)
    
    # origin_mask = resize_transform(origin_map)
    # erased_mask = resize_transform(erased_map)
    
    origin_score_map = overlay_mask(to_pil_image(origin_image[0]), to_pil_image(origin_map),)
    erased_score_map = overlay_mask(to_pil_image(erased_image[0]), to_pil_image(erased_map))
    
    pdb.set_trace()
    
    save_image(origin_score_map, 'origin.png')
    save_image(erased_score_map, 'erased.png')
    

if __name__ == "__main__":
    args = get_args_parser()
    visualize(args)
    
    