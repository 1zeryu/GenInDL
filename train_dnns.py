import torch
from datasets import *
from models import build_neural_network
from datasets import DatasetGenerator
from utils import *
from criterion import get_criterion
from torchvision import transforms
import numpy as np
from noise import *
import argparse
from torch.nn.utils import clip_grad_norm_

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

def train_one_epoch(net, optimizer, criterion, train_loader, args):
    # metrics
    acc_meter = AverageMeter()
    acc5_meter = AverageMeter()
    loss_meter = AverageMeter()
    
    # train
    net.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        logits = net(images)
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        
        if args.clip_grad: 
            clip_grad_norm_(net.parameters(), max_norm=args.clip_grad, norm_type=2)
        
        optimizer.step() 
        
        acc, acc5 = accuracy(logits, labels, topk=(1,5))
        acc_meter.update(acc)
        acc5_meter.update(acc5)
        loss_meter.update(loss.item())
    
        if i % args.log_frequency == 0:
            print("@Acc: {:.3f}, @Acc5: {:.3f}, @Loss: {:.3f}".format(acc, acc5, loss.item()))
    
    return acc_meter.avg, acc5_meter.avg, loss_meter.avg
    
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

def train_net_for_classification(net, optimizer, criterion, train_loader, eval_loader, lr_scheduler, args):
    print("Training network for classification")
    alpha_plan = [0.01] * 60 + [0.001] * 40

    for epoch in range(1, args.epochs):
        train_acc, train_acc5, train_loss = train_one_epoch(net, optimizer, criterion, train_loader, args)
        test_acc, test_acc5, test_loss = evaluate(net, criterion, eval_loader, args)
        
        print("""Iteration: [{:03d}/{:03d}]
              train: @acc: {:.3f}, @acc5: {:.3f}, @loss: {:.3f}
              test: @acc: {:.3f}, @acc5: {:.3f}, @loss: {:.3f}""".format(epoch, args.epochs, train_acc, train_acc5, train_loss, test_acc, test_acc5, test_loss))
        
        if args.lr_scheduler == 'alpha_plan':
            adjust_learning_rate(optimizer, alpha_plan[epoch])
        else:
            lr_scheduler.step()
        save_model(args.save, net, optimizer, args)
        
def train_dnns(args):
    # initialize 
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
        
    if torch.cuda.is_available():
        # # Automatically find the best optimization algorithm for the current configuration
        torch.backends.cudnn.enabled = True 
        torch.backends.cudnn.benchmark = True 
        
    # get the data loader
    data = DatasetGenerator(train_bs=args.batch_size, eval_bs=args.batch_size, n_workers=args.n_workers)
    train_loader, eval_loader = data.get_loader()
    
    # building the network
    net = build_neural_network(args.arch)
    net.to(device)
    optimizer = build_optimizer(args.optimizer, net, args.learning_rate, args)
    load_model(args.load, net, optimizer)
    
    # create optimizer 
    criterion = get_criterion(args.criterion, args.num_classes, args.confidence)
    lr_scheduler = build_lr_scheduler(args.lr_scheduler, optimizer)

    # net = build_perturbed_net(IBSAP(), net) # create the network with the noise input

    train_net_for_classification(net, optimizer, criterion, train_loader, eval_loader, lr_scheduler, args)
    
    
if __name__ == '__main__':
    args = get_args_parser()
    train_dnns(args)