from datetime import datetime 
import logging
import os
from tokenize import Name
import numpy as np
import torch
from scipy.spatial.distance import cdist
from torch.utils.tensorboard import SummaryWriter
import pdb
from thop import profile
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Params(model):
    return sum([param.nelement() for param in model.parameters()])


def FlopandParams(model):
    input = torch.randn(1, 3, 32, 32).to(device)
    flops, params = profile(model, inputs=(input,))
    return flops, params

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = max(self.max, val)


def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec

def count_parameters_in_MB(model):
    return sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary_head" not in name)/1e6

def rand_bbox(size, lam):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def setup_writer(name):
    writer = SummaryWriter(name)
    return writer

def track_loss(model, criterion, loader):
    loss_list = []
    correct_list = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if not isinstance(criterion, torch.nn.CrossEntropyLoss):
            logits, loss = criterion(model, images, labels, None)
        else:
            logits = model(images)
            loss = criterion(logits, labels)
        _, predicted = torch.max(logits.data, 1)
        b_correct = (predicted == labels)
        loss_list += loss.detach().cpu().numpy().tolist()
        correct_list += b_correct.detach().cpu().numpy().tolist()
    return loss_list, correct_list


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""
    formatter = logging.Formatter('%(asctime)s %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def log_display(epoch, global_step, time_elapse, **kwargs):
    display = 'epoch=' + str(epoch) + \
              ' global_step=' + str(global_step)
    for key, value in kwargs.items():
        if type(value) == str:
            display = ' ' + key + '=' + value
        else:
            display += ' ' + str(key) + '=%.4f' % value
    display += ' time=%.2fit/s' % (1. / time_elapse)
    return display


def build_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return

def debug():
    pdb.set_trace()

import time
class timer(object):
    def __init__(self) -> None:
        pass
    
    def logtime(self):
        return time.strftime('%a %b %d %H:%M:%S %Y', time.localtime())
    
    def filetime(self):
        return time.strftime("%yY%mM%dD%Hh%Mm%Ss", time.localtime())

    def get_hms(self, seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)

        return h, m, s

    start_time = 0
    epoch_time = 0
    running_time = 0
    def setStart(self):
        self.start_time = time.time()
    
    def setEpoch(self):
        self.epoch_time = time.time() - self.start_time
        self.running_time += self.epoch_time

    def runtime(self):
        return '| Running time : %d:%02d:%02d'  %(self.get_hms(self.running_time))

import matplotlib.pyplot as plt
def show_image(activation_map, name=None):
    plt.figure()
    plt.imshow(activation_map)
    plt.axis('off')
    plt.title(name)
    plt.tight_layout()
    
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)

    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(1/batch_size))
    return res
