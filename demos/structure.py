import torch
from torchvision.models import resnet18
from tensorboardX import SummaryWriter
from torchvision.io import read_image 


data = read_image('demo.jpg')

