import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss, _Loss
from utils.misc import *

class LogitLoss(_WeightedLoss):
    def __init__(self, num_classes):
        super(LogitLoss, self).__init__()
        self.num_classes = num_classes
        self.use_cuda = torch.cuda.is_available()

    def forward(self, input, target):
        one_hot_labels = one_hot(target.cpu(), num_classes=self.num_classes)
        if self.use_cuda:
            one_hot_labels = one_hot_labels.cuda()

        # Get the logit output value
        logits = (one_hot_labels * input).max(1)[0]
        # Increase the logit value
        return torch.mean(-logits)

class BoundedLogitLoss(_WeightedLoss):
    def __init__(self, num_classes, confidence=-10):
        super(BoundedLogitLoss, self).__init__()
        self.num_classes = num_classes
        self.confidence = confidence
        self.use_cuda = torch.cuda.is_available()

    def forward(self, input, target):
        one_hot_labels = one_hot(target.cpu(), num_classes=self.num_classes)
        if self.use_cuda:
            one_hot_labels = one_hot_labels.cuda()

        target_logits = (one_hot_labels * input).sum(1)
        not_target_logits = ((1. - one_hot_labels) * input - one_hot_labels * 10000.).max(1)[0]
        logit_loss = torch.clamp(not_target_logits - target_logits, min=-self.confidence)
        return torch.mean(logit_loss)

class BoundedLogitLossFixedRef(_WeightedLoss):
    def __init__(self, num_classes, confidence=-10):
        super(BoundedLogitLossFixedRef, self).__init__()
        self.num_classes = num_classes
        self.confidence = confidence
        self.use_cuda = torch.cuda.is_available()

    def forward(self, input, target):
        one_hot_labels = one_hot(target.cpu(), num_classes=self.num_classes)
        if self.use_cuda:
            one_hot_labels = one_hot_labels.cuda()

        target_logits = (one_hot_labels * input).sum(1)
        not_target_logits = ((1. - one_hot_labels) * input - one_hot_labels * 10000.).max(1)[0]
        logit_loss = torch.clamp(not_target_logits.data.detach() - target_logits, min=-self.confidence)
        return torch.mean(logit_loss)

class BoundedLogitLoss_neg(_WeightedLoss):
    def __init__(self, num_classes, confidence=-10):
        super(BoundedLogitLoss_neg, self).__init__()
        self.num_classes = num_classes
        self.confidence = confidence
        self.use_cuda = torch.cuda.is_available()

    def forward(self, input, target):
        one_hot_labels = one_hot(target.cpu(), num_classes=self.num_classes)
        if self.use_cuda:
            one_hot_labels = one_hot_labels.cuda()

        target_logits = (one_hot_labels * input).sum(1)

        not_target_logits = ((1. - one_hot_labels) * input - one_hot_labels * 10000.).max(1)[0]
        logit_loss = torch.clamp(target_logits - not_target_logits, min=-self.confidence)
        return torch.mean(logit_loss)
    
class MaxMapLoss(_WeightedLoss):
    def __init__(self):
        super(MaxMapLoss, self).__init__()
    
    def forward(self, input, target):
        B = input.shape[0]
        max_sum = 0
        for i in range(B):
            max_sum += torch.max(torch.abs(input - target))
        return max_sum / B
    
loss_func_lst = {
    'logitloss': lambda kwargs: LogitLoss(kwargs['num_classes']),
    'boundedlogitloss': lambda kwargs: BoundedLogitLoss(kwargs['num_classes'], kwargs['confidence']),
    'Boundedlogitlossfixedref': lambda kwargs: BoundedLogitLossFixedRef(kwargs['num_classes'], kwargs['confidence']),
    'Boundedlogitloss_neg': lambda kwargs: BoundedLogitLoss_neg(kwargs['num_classes'], kwargs['confidence']),
    'crossentropyloss': lambda kwargs: torch.nn.CrossEntropyLoss(),
    'l1loss': lambda kwargs: torch.nn.L1Loss(),
    'smoothl1loss': lambda kwargs: torch.nn.SmoothL1Loss(),
    'mseloss': lambda kwargs: torch.nn.MSELoss(),
}    

def get_criterion(loss_func, num_classes=None, confidence=0):
    """Return the criterion for training the model

    Args:
        loss_func (str): the name of the loss function

    Returns:
        loss_function (_WeightedLoss): the Pytorch loss function
    """
    kwargs = {
        'num_classes': num_classes,
        'confidence': confidence,
    }
    return loss_func_lst[loss_func](kwargs)
