import mlconfig
import torch
import torch.nn as nn
import torchvision
from . import DenseNet, ResNet, ToyModel, vgg, alexnet, wrn, ViT

mlconfig.register(torch.optim.SGD)
mlconfig.register(torch.optim.Adam)
mlconfig.register(torch.optim.lr_scheduler.MultiStepLR)
mlconfig.register(torch.optim.lr_scheduler.CosineAnnealingLR)
mlconfig.register(torch.optim.lr_scheduler.StepLR)
mlconfig.register(torch.optim.lr_scheduler.ExponentialLR)
mlconfig.register(torch.nn.CrossEntropyLoss)

# Models
mlconfig.register(ResNet.ResNet)
mlconfig.register(ResNet.ResNet18)
mlconfig.register(ResNet.ResNet34)
mlconfig.register(ResNet.ResNet50)
mlconfig.register(ResNet.ResNet101)
mlconfig.register(ResNet.ResNet152)
mlconfig.register(ToyModel.ToyModel)
mlconfig.register(DenseNet.DenseNet121)
mlconfig.register(vgg.VGG)
mlconfig.register(alexnet.AlexNet)
mlconfig.register(wrn.RobustWideResNet)
mlconfig.register(ViT.VisionTransformer)
mlconfig.register(alexnet.AlexNet_fine_tuning)

# torchvision models
mlconfig.register(torchvision.models.resnet18)
mlconfig.register(torchvision.models.resnet50)
mlconfig.register(torchvision.models.densenet121)
mlconfig.register(torchvision.models.vgg11_bn)
mlconfig.register(torchvision.models.ResNet18_Weights)