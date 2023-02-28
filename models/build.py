from .alexnet import *
from .vgg import *
from .DenseNet import *
from .ResNet import *

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
}

def build_neural_network(arch):
    return __model_dict[arch](num_classes=10)