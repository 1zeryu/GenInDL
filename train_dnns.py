import torch
from datasets import *
from models import build_neural_network
from datasets.dataset import DatasetGenerator
from utils import *
from criterion import get_criterion
from torchvision import transforms
import numpy as np
import argparse
from torch.nn.utils import clip_grad_norm_

# os.environ["WANDB_MODE"] = "disabled"
# os.environ["WANDB_SILENT"] = "true"
os.environ['WANDB_API_KEY'] = 'ec5d114180c22f7ec57e35cf5d7370f4c6ffe839'
import wandb
wandb.login()

def get_args_parser():
    # get the parameters from the terminal
    parser = argparse.ArgumentParser(description='Generalization in Deep Learning')
    parser.add_argument('--task', type=str, default='train_net_for_classification')
    # Experiment Options

    # system parameters
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_frequency', type=int, default=50)

    # data parameters 
    parser.add_argument('--batch_size', type=int, default=256, ) 
    parser.add_argument('--n_workers', type=int, default=4)
    
    # neural network parameters
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--save', type=str, default=None)
    
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
    parser.add_argument("--notes", type=str, default=None)
    
    parser.add_argument('--beta', default=1, type=float,
                        help='hyperparameter beta')
    parser.add_argument('--cut_prob', default=0, type=float,
                        help='cutmix probability')
    
    args = parser.parse_args()
    return args

beta_t = [1] * 40 + [0.5] * 60
from torchvision.utils import make_grid
def train_one_epoch(net, optimizer, criterion, train_loader, args, epoch):
    # metrics
    acc_meter = AverageMeter()
    acc5_meter = AverageMeter()
    loss_meter = AverageMeter()
    
    # train
    net.train()
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda()
        target = target.cuda()

        r = np.random.rand(1)
        
        # if args.beta > 0 and r < args.cutmix_prob:
        #     # generate mixed sample
        #     lam = np.random.beta(args.beta, args.beta)
        #     rand_index = torch.randperm(input.size()[0]).cuda()
        #     target_a = target
        #     target_b = target[rand_index]
        #     bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
        #     input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
        #     # adjust lambda to exactly match pixel ratio
        #     lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
        #     # compute output
        #     output = model(input)
        #     loss = criterion(output, target_a) * lam + criterion(output, target_b) * (1. - lam)
        
        if args.beta > 0 and r < args.cut_prob:
            # generate mixed sample
            # pdb.set_trace()
            lam = np.random.uniform(0, args.beta)
            
            rand_index = torch.randperm(input.size()[0]).cuda()
            target_a = target
            target_b = target[rand_index]
            
            height = 32
            width = 32 
            n_channels = 3
            
            rand = torch.rand(height, width)
            masks = (rand < lam).float().cuda()
            
            input[:, :] =  input[:, :] * (1 - masks) # + input[rand_index, :]  * masks 
            
            output = net(input)
            loss = criterion(output, target_a) 
            
        
        else:
            # compute output
            lam = 0
            output = net(input)
            loss = criterion(output, target)
            
        if epoch == 1 and i < 5:
            input_data = wandb.Table(columns=['Image', 'lam'])
            image = wandb.Image(make_grid(input, 16))
            input_data.add_data(image, lam)            
                
        optimizer.zero_grad()
        loss.backward()
        
        if args.clip_grad: 
            clip_grad_norm_(net.parameters(), max_norm=args.clip_grad, norm_type=2)
        
        optimizer.step() 
        
        acc, acc5 = accuracy(output, target, topk=(1,5))
        acc_meter.update(acc)
        acc5_meter.update(acc5)
        loss_meter.update(loss.item())
    
        if i % args.log_frequency == 0:
            print("@Acc: {:.3f}, @Acc5: {:.3f}, @Loss: {:.3f}".format(acc, acc5, loss.item()))
            
    # push the input image
    # pdb.set_trace()
    args.beta = beta_t[epoch]
    
    if epoch == 1:
        wandb.log(input_data)
    
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

import datetime
def train_net_for_classification(net, optimizer, criterion, train_loader, eval_loader, lr_scheduler, args):
    print("Training network for classification")
    alpha_plan = [0.01] * 60 + [0.001] * 40
    
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    run = wandb.init(project='sparsemix', name=nowtime, notes=args.notes, save_code=True)
    wandb.config = {
        'batch_size': args.batch_size,
        'arch': args.arch,
        'load': args.load,
        'save': args.save,
        'criterion': args.criterion,
        'lr_scheduler': args.lr_scheduler,
        'optimizer': args.optimizer,
        'learning_rate': args.learning_rate,
        'beta': args.beta,
        'cut_prob': args.cut_prob, 
        'epochs': args.epochs,
    }

    for epoch in range(1, args.epochs):
        train_acc, train_acc5, train_loss = train_one_epoch(net, optimizer, criterion, train_loader, args, epoch)
        test_acc, test_acc5, test_loss = evaluate(net, criterion, eval_loader, args)
        
        print("""Iteration: [{:03d}/{:03d}]
              train: @acc: {:.3f}, @acc5: {:.3f}, @loss: {:.3f}
              test: @acc: {:.3f}, @acc5: {:.3f}, @loss: {:.3f}""".format(epoch, args.epochs, train_acc, train_acc5, train_loss, test_acc, test_acc5, test_loss))
        
        wandb.log({'train_loss': train_loss, 'train_acc': train_acc,'top1': test_acc})
        
        if args.lr_scheduler == 'alpha_plan':
            adjust_learning_rate(optimizer, alpha_plan[epoch])
        else:
            lr_scheduler.step()
            
        if args.save != None:
            save_model(args.save, net, optimizer, args)
        
    code = wandb.Artifact('python', type='code')
    code.add_file('train_dnns.py')
    wandb.log_artifact(code)
    
    wandb.finish()
        
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
    if args.load != None:
        load_model(args.load, net, optimizer)
    
    # create optimizer 
    criterion = get_criterion(args.criterion, args.num_classes, args.confidence)
    lr_scheduler = build_lr_scheduler(args.lr_scheduler, optimizer)

    # net = build_perturbed_net(IBSAP(), net) # create the network with the noise input

    train_net_for_classification(net, optimizer, criterion, train_loader, eval_loader, lr_scheduler, args)
    
    
if __name__ == '__main__':
    args = get_args_parser()
    train_dnns(args)