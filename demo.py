from datasets.dataset import DatasetGenerator
import torch
from torchvision.utils import make_grid, save_image
import torchattacks
from utils import accuracy, AverageMeter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os

def train_targets_map(images, labels):
    return labels

plot_or_not = False

def GenerateMask(loader, model):
    atk = torchattacks.PGDL2(model)
    atk.set_mode_targeted_by_function(train_targets_map)
    acc_meters = AverageMeter()
    acc5_meters = AverageMeter()
    loss_meters = AverageMeter()
    for i, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        adv_images = atk(images, labels)
        noise = adv_images - images
        save_dir = 'imgs'
        if plot_or_not:
            save_image(make_grid(adv_images), os.path.join(save_dir, 'adversarial_images{}.jpg'.format(str(i))))
        
        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                logits = model(images)
                loss = criterion(logits, labels)
        
        loss = loss.item()
        acc, acc5 = accuracy(logits, labels, topk=(1, 5))
        print('acc: {:.4f}, acc5: {:.4f}, loss: {:.4f}'.format(acc, acc5, loss))
        acc_meters.update(acc.item())
        acc5_meters.update(acc5.item())
        loss_meters.update(loss)
    print("Average: acc: {:.4f}, acc5: {:.4f}, loss: {:.4f}".format(acc_meters.avg, acc5_meters.avg, loss_meters.avg))
    