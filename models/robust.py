from .ResNet import *
import torch


class Kireev2021EffectivenessNet(PreActResNet):

    def __init__(self, num_classes=10):
        super(Kireev2021EffectivenessNet, self).__init__(PreActBlockV2,
                                                         [2, 2, 2, 2],
                                                         bn_before_fc=True)
        self.register_buffer(
            'mu',
            torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1))
        self.register_buffer(
            'sigma',
            torch.tensor([0.2471, 0.2435, 0.2616]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super(Kireev2021EffectivenessNet, self).forward(x)
    

class Modas2021PRIMEResNet18(ResNet):

    def __init__(self, num_classes=10):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

        # mu & sigma are updated from weights
        self.register_buffer('mu', torch.tensor([0.5] * 3).view(1, 3, 1, 1))
        self.register_buffer('sigma', torch.tensor([0.5] * 3).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mu) / self.sigma
        return super().forward(x)
    
