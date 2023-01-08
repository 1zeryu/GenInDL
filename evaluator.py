from unittest import loader
import torch
from utils import AverageMeter, accuracy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Evaluator():
    def __init__(self, criterion, loader, logger):
        self.loader = loader
        self.criterion = criterion
        self.logger = logger
        self.loss_meters = AverageMeter()
        self.acc_meters = AverageMeter()
        self.acc5_meters = AverageMeter()

    def _reset_stats(self):
        self.loss_meters = AverageMeter()
        self.acc_meters = AverageMeter()
        self.acc5_meters = AverageMeter()

    def eval(self, model, exp_stats={}):
        self._reset_stats()
        model.eval()
        for i, (images, labels) in enumerate(self.loader):
            self.eval_batch(images=images, labels=labels, model=model)
        payload = 'Val Loss: %.2f' % self.loss_meters.avg
        self.logger.info('\033[33m'+payload+'\033[0m')
        exp_stats['val_loss'] = self.loss_meters.avg
        payload = 'Val Acc: %.4f' % self.acc_meters.avg
        self.logger.info('\033[33m'+payload+'\033[0m')
        exp_stats['val_acc'] = self.acc_meters.avg
        return exp_stats

    def eval_batch(self, images, labels, model):
        images.requires_grad(True)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.no_grad():
            if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
                logits = model(images)
                loss = self.criterion(logits, labels)
            else:
                logits, loss = self.criterion(model, images, labels, None)
        loss = loss.item()
        self.loss_meters.update(loss, images.shape[0])
        acc, acc5 = accuracy(logits, labels, topk=(1, 5))
        self.acc_meters.update(acc.item(), labels.shape[0])
        self.acc5_meters.update(acc5.item(), labels.shape[0])
        return loss
    
