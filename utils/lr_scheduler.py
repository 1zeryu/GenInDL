from torch.optim.lr_scheduler import *


lr_scheduler_lst = {
    ''
}

def build_lr_scheduler(lr_duler, optimizer, gamma=0.1, step_size=5, last_epoch=-1, eta_min=0 ,T_max=10):
    """building lr_scheduler for optimization

    Args:
        lr_duler (str): lr_scheduler name
        optimizer (optim): Pytorch optimizer
    """
    if lr_duler == 'step':
        return StepLR(optimizer, step_size=step_size, gamma=0.1, last_epoch=last_epoch)
    
    elif lr_duler == 'exponential':
        return ExponentialLR(optimizer, gamma=gamma, last_epoch=last_epoch)
    
    elif lr_duler == 'cosineannealing':
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min, last_epoch=last_epoch)
    

def adjust_learning_rate(optimizer, lr):
    """adjust learning rate 

    Args:
        optimizer (nn.optim): the optimizer 
        lr (float): the next epoch learning rate 
    """
    for param in optimizer.param_groups:
        param['lr'] = lr