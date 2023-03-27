import torch
from torchvision import transforms
import os
import torchvision.datasets as datasets 
from torchvision import models
import argparse
from criterion import *
from utils import *
from torchcam.methods import SmoothGradCAMpp
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.transforms.functional import to_pil_image
import random 

def get_args():
    parser = argparse.ArgumentParser(description='Generalization in Imagenet')
    parser.add_argument('--batch_size', type=int, default=256,)
    parser.add_argument('--n_workers', type=int, default=4)
    
    parser
    parser.add_argument('--data_dir', type=str, default='/root/autodl-tmp/imagenet')
    parser.add_argument('--arch', type=str, default='resnet18')
    args = parser.parse_args()
    return args

class Eraser(object):
    def __init__(self, erasing_method, erasing_ratio, model):
        self.erasing_method = erasing_method
        self.erasing_ratio = erasing_ratio
        self.model = model
        
        self.cam_extractor = SmoothGradCAMpp(model, target_layer='layer4.1.conv2')
        
    def global_random_erasing(self, img):    
        # get gaussian erasing image 
        process_img = img.clone()
        finish = torch.zeros_like(img).to(device)
        
        # CIFAR-N image shape
        HW = 224 * 224
        salient_order = torch.randperm(HW).to(device).reshape(1, -1)
        coords = salient_order[:, 0: int(HW * self.erasing_ratio)]
        process_img.reshape(1, 3, HW)[0, :, coords] = finish.reshape(1, 3, HW)[0, :, coords]
        return process_img
    
    def front_random_erasing(self, img):
        process_img = img.clone()
        out = self.model(img.unsqueeze(0))
        map = self.cam_extractor(class_idx=out.squeeze(0).argmax().item(), scores=out)[0]
        erasing_map = self.map_tool(to_pil_image(map))
        finish = torch.zeros_like(img).to(device)
        
        # CIFAR-N image shape
        HW = 224 * 224
        salient_order = torch.flip(torch.argsort(erasing_map.reshape(-1, HW), dim=1), dims=[1]).to(device)
        coords = salient_order[:, 0:int(HW*0.5)]
        shuffled_coords = coords[:, torch.randperm(coords.size(1))]
        
        random_flip_coords = shuffled_coords[:,:int(HW *self.erasing_ratio)]
        process_img.reshape(1, 3, HW)[0, :, random_flip_coords] = finish.reshape(1, 3, HW)[0, :, random_flip_coords]
        return process_img
        
    
    def back_random_erasing(self, img):
        process_img = img.clone()
        out = self.model(img.unsqueeze(0))
        map = self.cam_extractor(class_idx=out.squeeze(0).argmax().item(), scores=out)[0]
        erasing_map = self.map_tool(to_pil_image(map))
        finish = torch.zeros_like(img).to(device)
        
        # CIFAR-N image shape
        HW = 224 * 224
        salient_order = torch.argsort(erasing_map.reshape(-1, HW), dim=1).to(device)
        coords = salient_order[:, 0:int(HW*0.5)]
        shuffled_coords = coords[:, torch.randperm(coords.size(1))]
        
        random_flip_coords = shuffled_coords[:,:int(HW *self.erasing_ratio)]
        process_img.reshape(1, 3, HW)[0, :, random_flip_coords] = finish.reshape(1, 3, HW)[0, :, random_flip_coords]
        return process_img
        
        
    def __call__(self, img, label):
        if self.erasing_method == 'global_random_erasing':
            return self.global_gaussian(img)
        
        elif self.erasing_method == 'front_random_erasing':
            return self.front_random_erasing(img)
        
        elif self.erasing_method == 'back_random_erasing':
            return self.back_random_erasing(img)

        
    
    

from tqdm import tqdm
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
    eraser = Eraser(erasing_method, erasing_ratio, model)
    process_dataset = []
    for i, (images, targets) in tqdm(enumerate(loader), desc=desc):
        images = images.to(device)
        targets = targets.to(device) 
        for image, label in zip(images, targets):
            process_dataset.append(eraser(image, label))
    dataset = torch.stack(process_dataset)
    return dataset

# get the data of imagenet 
def get_imagenet_data(data_dir, batch_size, num_workers, pin_memory, distributed=False):
    traindir = os.path.join(data_dir, 'train')
    valdir = os.path.join(data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    test_dataset = datasets.ImageFolder(
        valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    test_dataset = torch.utils.data.Subset(test_dataset, range(1000))
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, sampler=test_sampler)
    return val_loader

def get_neural_network(arch):
    net_builder = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'vit_b_16': models.vit_b_16,
        'vit_b_32': models.vit_b_32,
        'wide_resnet50_2': models.wide_resnet50_2,
        'wide_resnet101_2': models.wide_resnet101_2,
    }
    
    return net_builder[arch](pretrained=True)

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

def ImageNet(args):
    val_loader = get_imagenet_data(args.data_dir, args.batch_size, args.n_workers, True)
    
    # pdb.set_trace()
    
    net = get_neural_network(args.arch)
    criterion = get_criterion('crossentropyloss')
    
    acc, acc5, loss = evaluate(net, criterion, val_loader, args)
    print(f"acc: {acc}, acc5: {acc5}, loss: {loss}")
    
if __name__ == "__main__":
    args = get_args()
    ImageNet(args)