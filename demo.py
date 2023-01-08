from datasets.dataset import DatasetGenerator
import torch
from torchvision.utils import make_grid, save_image
from functions import pgd_attack
from utils import accuracy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os

def GenerateMask(loader, model, criterion=None):
    for i, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        adv_images = pgd_attack(model, images, labels)
        noise = adv_images - images
        
        save_dir = 'imgs'
        save_image(make_grid(noise), os.path.join(save_dir, 'noise{}.jpg'.format(str(i))))
        
        with torch.no_grad():
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                logits = model(adv_images)
                loss = criterion(logits, labels)
        
        loss = loss.item()
        acc, acc5 = accuracy(logits, labels, topk=(1, 5))
        print('acc: {:.4f}, acc5: {:.4f}, loss: {:.4f}'.format(acc, acc5, loss))
    
    