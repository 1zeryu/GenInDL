import torch.optim as optim

# __optimizer_dict = {
#     'sgd': optim.SGD,
#     'adam': optim.Adam,
#     'rmsprop': optim.RMSprop,
#     'adamw': optim.AdamW,
#     'l-bfgs': optim.LBFGS,
#     'nadam': optim.NAdam,
# }

def build_optimizer(arch, model, lr, args):
    """build an optimizer for the given model and architecture

    Args:
        arch (str): optimizer architecture
        model (nn.Module): the Pytorch model 
        lr (float): learning rate for the optimizer
        (Optional)
        
    """
    if arch == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)

    elif arch == 'adam':
        return optim.Adam(model.parameters(), lr=lr, betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay)
    
    elif arch == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=lr, momentum=args.momentum, alpha=args.alpha, weight_decay=args.weight_decay)
    
    elif arch == 'adamw':
        return optim.AdamW(model.parameters())