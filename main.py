import argparse
from ast import Store
from email.headerregistry import Group
from email.policy import default
from pydoc import cram
from unittest import loader 
import torch
import numpy as np
from datasets.dataset import DatasetGenerator
from evaluator import Evaluator
from exp_mgnt import ExperimentManager
from trainer import Trainer 
from utils import *
from cam import GroupCAM, groupcam
import torch
from torchvision.io import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torch.utils.tensorboard import SummaryWriter

if torch.cuda.is_available():
    # # Automatically find the best optimization algorithm for the current configuration
    torch.backends.cudnn.enabled = True 
    torch.backends.cudnn.benchmark = True 
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

exp_configs = 'configs'

# get the parameters from the terminal
parser = argparse.ArgumentParser(description='Generalization in Deep Learning')
# Experiment Options
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--exp', type=str, default='rn18_cifar10')
parser.add_argument('--if_logger', '-l', default=False, action='store_true')
parser.add_argument('--if_writer','-w', default=False, action='store_true')
parser.add_argument('--load_model', default=None, type=str)
parser.add_argument('--only_test', '-o', action='store_true', default=False)
parser.add_argument('--save', '-s', action='store_true', default=False)
parser.add_argument('--cam','-c', action='store_true', default=False)
args = parser.parse_args()



# extractor the units from the datasets
def cam(model, loader, target_layer='conv1'):
    gc = GroupCAM(model, target_layer=target_layer)
    for i, (data, target) in enumerate(loader):
        if torch.cuda.is_available():
            data = data.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        saliency_maps = []
        model.eval()
        for idx in range(data.shape[0]):
            image = data[idx].unsqueeze(0)
            saliency = gc(image, class_idx=target[idx], retain_graph=True)
            saliency = saliency.to(image.device)
            saliency_maps.append(saliency)
        debug()
        saliency_maps = torch.cat(saliency_maps, dim=0)
        mean = torch.mean(saliency_maps)
        saliency_maps = torch.where(saliency_maps < mean, 0.0, 1.0)

def main():
    exp = ExperimentManager(args)
    if args.load_model is not None:
        model = exp.load(args.load_model)
    model = exp.model
    optimizer = exp.optimizer
    criterion = exp.criterion
    data = exp.data
    train_loader, eval_loader, posion_loader = data.get_loader()
    scheduler = exp.scheduler
    start_epoch = 0
    best_acc = exp.best_acc
    
    evaluator = Evaluator(criterion=criterion, loader=eval_loader, logger=exp)
    trainer = Trainer(criterion=criterion, loader=train_loader, exp=exp)
    epochs = exp.epoch
    
    if args.only_test:
        exp.info('Only Test')
        exp_stats = {}
        model.eval()
        exp_stats = evaluator.eval(model, exp_stats=exp_stats)
        exp.info(exp_stats)
        exit(0)
    
    for epoch in range(start_epoch, epochs):
        exp_stats = {} # is a directory containing ['lr', 'global_step', 'train_loss', 'train_acc', 'val_loss', 'val_acc']
        exp.info("="*20 + "Training Epoch %d" % (epoch) + "="*20)
        model.train()
        exp_stats = trainer.train(epoch, model, optimizer, exp_stats=exp_stats)
        exp.info("="*20 + "Eval Epoch %d" % (epoch) + "="*20)
        model.eval()
        exp_stats = evaluator.eval(model, exp_stats=exp_stats)
        
        exp.write('acc/train', exp_stats['train_acc'], epoch)
        exp.write('acc/eval', exp_stats['val_acc'], epoch)
        exp.write('loss/train', exp_stats['train_loss'], epoch)
        exp.write('loss/eval', exp_stats['val_loss'], epoch)
        
        if exp_stats['val_acc'] > best_acc:
            state = {
                'model': model.state_dict(),
                'best_acc': exp_stats['val_acc'],
                'train_loss': exp_stats['val_loss'],
                'eval_loss': exp_stats['val_loss'],
                'train_acc': exp_stats['train_acc'],
            }
            exp.save(state)
            best_acc = exp_stats['val_acc']
            exp.info('!' * 10 + 'The best accuracy is {}'.format(best_acc) + '!' * 10)
            
    return    


if __name__ == '__main__':
    main()
    

