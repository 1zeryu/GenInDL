from .alexnet import *
from .vgg import *
from .DenseNet import *
from .robust import *
from .ResNet import *
from .resnetx import *

__model_dict = {
    'alexnet': AlexNet,
    
    'resnet18': ResNet18,
    'resnet34': ResNet34,
    'resnet50': ResNet50,
    'resnet101': ResNet101,
    'resnet152': ResNet152,
    
    'vgg11': vgg11,
    'vgg13': vgg13,
    'vgg16': vgg16,
    'vgg19': vgg19,
    
    'densenet121': DenseNet121,
    'densenet169': DenseNet169,
    'densenet201': DenseNet201,
    'densenet161': DenseNet161,
    
    'Kireev2021EffectivenessNet': Kireev2021EffectivenessNet,
    'Modas2021PRIMEResNet18': Modas2021PRIMEResNet18,
    
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'resnet110': resnet110,
    'resnet1202': resnet1202,
}

def build_neural_network(arch):
    return __model_dict[arch](num_classes=10)