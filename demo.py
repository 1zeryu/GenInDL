from torchvision import models

from utils.lid import structure
import torch
model = models.resnet18()

import utils

structure(model, torch.rand(1, 3, 224, 224), 'resnet18')