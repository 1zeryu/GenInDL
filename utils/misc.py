import os
from tokenize import Name
import numpy as np
import torch
import pdb
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Params(model):
    return sum([param.nelement() for param in model.parameters()])

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

def one_hot(class_labels, num_classes=None):
    if num_classes==None:
        return torch.zeros(len(class_labels), class_labels.max()+1).scatter_(1, class_labels.unsqueeze(1), 1.)
    else:
        return torch.zeros(len(class_labels), num_classes).scatter_(1, class_labels.unsqueeze(1), 1.)

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


def build_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return

def debug():
    pdb.set_trace()
    

def save_model(name, net, optimizer, args=None):
    model_state_dict_path = 'experiments/state_dict/'
    if os.path.exists(model_state_dict_path):
        state = {
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'args': args,
        }
        file_name = name + '.pt'
        path = os.path.join(model_state_dict_path, file_name)
        torch.save(state, path)
        print("Model saved to %s" % path)

def load_model(name, net, optimizer):
    model_state_dict_path = os.path.join('experiments/state_dict', name + '.pt')
    if os.path.exists(model_state_dict_path):
        state = torch.load(model_state_dict_path)
        net.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        print("Model loaded")
        return 1
    else:
        print("Model Pt not found")
        return 0

from collections import OrderedDict
def build_perturbed_net(generator, target_model):
    perturbed_net = torch.nn.Sequential(OrderedDict([('generator', generator), ('target_model', target_model)]))
    if torch.cuda.is_available():
        perturbed_net.cuda()
    return perturbed_net

