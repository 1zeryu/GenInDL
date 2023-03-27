import torch
from torchvision import transforms
import os
import torchvision.datasets as datasets 
from torchvision import models
import argparse
from criterion import *
from utils import *

def get_args():
    parser = argparse.ArgumentParser(description='Generalization in Imagenet')
    parser.add_argument('--batch_size', type=int, default=256,)
    parser.add_argument('--n_workers', type=int, default=4)
    
    parser
    parser.add_argument('--data_dir', type=str, default='/root/autodl-tmp/imagenet')
    parser.add_argument('--arch', type=str, default='resnet18')
    args = parser.parse_args()
    return args

# get the data of imagenet 
def get_imagenet_data(data_dir, batch_size, num_workers, pin_memory, distributed=False):
    traindir = os.path.join(data_dir, 'train')
    valdir = os.path.join(data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # train_dataset = datasets.ImageFolder(
    #     traindir,
    #     transforms.Compose([
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]))
    # if distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # else:
    #     train_sampler = None
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
    #     num_workers=num_workers, pin_memory=pin_memory, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory)
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