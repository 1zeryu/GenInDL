from torchvision.models import alexnet, AlexNet_Weights

from models.ResNet import ResNet34
from models.vgg import VGG

def structure(model):
    model.eval()
    for module in model.named_modules():
        print(module[0])
        
model = ResNet34()

structure(model)