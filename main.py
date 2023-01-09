import torch
import models
import numpy as np
from datasets.dataset import DatasetGenerator
from evaluator import Evaluator
from exp_mgnt import ExperimentManager
from trainer import Trainer 
from utils import *
import torch
from torchvision.io import read_image
import random
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
if torch.cuda.is_available():
    # # Automatically find the best optimization algorithm for the current configuration
    torch.backends.cudnn.enabled = True 
    torch.backends.cudnn.benchmark = True 
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

exp_configs = 'configs'
import argparse

# get the parameters from the terminal
parser = argparse.ArgumentParser(description='Generalization in Deep Learning')
# Experiment Options
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--exp', type=str, default='rn18_cifar10')
parser.add_argument('--if_logger', '-l', default=False, action='store_true')
parser.add_argument('--writer','-w', default=False, action='store_true')
parser.add_argument('--load_model', default=None, type=str)
parser.add_argument('--only_test', '-o', action='store_true', default=False)
parser.add_argument('--cam','-c', default=False, action='store_true')
parser.add_argument('--alpha', type=float, default=0.02)
parser.add_argument('--save', action='store_true', default=False)
parser.add_argument('--noise_type', type=str, default='deletion',)
parser.add_argument('--demo', '-d', action='store_true', default=False)
args = parser.parse_args()

from mask import MaskDatasetGenerator

def main():
    exp = ExperimentManager(args)
    model = exp.model
    best_acc = 0
    if args.load_model is not None:
        state = exp.load(args.load_model)
        model.load_state_dict(state['model'])
        best_acc = state['best_acc']
        exp.info("!!! The best accuracy is {} !!!".format(best_acc))
    optimizer = exp.optimizer
    criterion = exp.criterion
    data = exp.data
    if args.save:
        train_loader, eval_loader = data.get_loader(train_shuffle=False)
    else:
        train_loader, eval_loader = data.get_loader(train_shuffle=True)
    scheduler = exp.scheduler
    start_epoch = 0
    if args.cam and args.alpha != 0:
        print("Masking...")
        if not args.only_test:
            train_loader = MaskDatasetGenerator(train_loader, eval_loader, model, 
                                            args.alpha, exp=exp, 
                                            type=args.noise_type, download=args.save, dataset=exp.data.dataset, train=True)
        else: 
            eval_loader = MaskDatasetGenerator(train_loader, eval_loader, model, args.alpha, exp=exp,
                                               type=args.noise_type, download=args.save, dataset=exp.data.dataset, train=False)    
    
    evaluator = Evaluator(criterion=criterion, loader=eval_loader, logger=exp)
    trainer = Trainer(criterion=criterion, loader=train_loader, exp=exp)
    epochs = exp.epoch
    
    if args.only_test:
        exp.info('Only Test')
        exp_stats = {}
        model.eval()
        exp_stats = evaluator.eval(model, exp_stats=exp_stats)
        exp.info(exp_stats)
    
    else:
        for epoch in range(start_epoch, epochs):
            exp_stats = {} # is a directory containing ['lr', 'global_step', 'train_loss', 'train_acc', 'val_loss', 'val_acc']
            exp.info("="*20 + "Training Epoch %d" % (epoch) + "="*20)
            model.train()
            exp_stats = trainer.train(epoch, model, optimizer, exp_stats=exp_stats)
            scheduler.step()
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
                path = exp.save(state, name = 'state_dict' + 'alpha' + str(args.alpha))
                best_acc = exp_stats['val_acc']
                exp.info('=' * 10 + 'The best accuracy is {}'.format(best_acc) + '=' * 10)

import warnings

if __name__ == '__main__':
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    main()
    

