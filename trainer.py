from asyncio.log import logger
import torch
from utils import AverageMeter, log_display, accuracy
from utils.exp import AverageMeter
import time
from torch.nn.utils import clip_grad_norm_

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Trainer():
    def __init__(self, criterion, loader, exp, global_step=0):
        self.loader = loader
        self.criterion = criterion
        self.logger = exp
        self.log_frequency = exp.log_frequency
        self.grad_clip = exp.grad_clip
        
        self.loss_meters = AverageMeter()
        self.acc_meters = AverageMeter()
        self.acc5_meters = AverageMeter()
        self.global_step = global_step

    def _reset_stats(self):
        self.loss_meters = AverageMeter()
        self.acc_meters = AverageMeter()
        self.acc5_meters = AverageMeter()

    def train(self, epoch, model, optimizer, exp_stats={}):
        self._reset_stats()
        for i, data in enumerate(self.loader):
            images, labels = data
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            start = time.time()
            log_payload = self.train_batch(images, labels, model, optimizer)
            end = time.time()
            time_used = end - start
            if self.global_step % self.log_frequency == 0:
                display = log_display(epoch=epoch,
                                           global_step=self.global_step,
                                           time_elapse=time_used,
                                           **log_payload)
                self.logger.info(display)
            self.global_step += 1
        exp_stats['global_step'] = self.global_step
        exp_stats['lr'] = optimizer.param_groups[0]['lr']
        exp_stats['train_loss'] = self.loss_meters.avg

        exp_stats['train_acc'] = self.acc_meters.avg
        return exp_stats

    def train_batch(self, images, labels, model, optimizer):
        model.zero_grad()
        optimizer.zero_grad()
        images = images.to(device)
        labels = labels.to(device)
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            logits = model(images)
            loss = self.criterion(logits, labels)
        else:
            logits, loss = self.criterion(model, images, labels, optimizer)
        loss.backward()
        if self.grad_clip != -1:
            grad_norm = clip_grad_norm_(model.parameters(),
                                        self.config.grad_clip)
        else:
            grad_norm = 0
        optimizer.step()

        self.loss_meters.update(loss.item(), labels.shape[0])
        payload = {"loss": loss,
                   "loss_avg": self.loss_meters.avg,
                   "lr": optimizer.param_groups[0]['lr'],
                   "|gn|": grad_norm}

        acc, acc5 = accuracy(logits, labels, topk=(1, 5))
        self.acc_meters.update(acc.item(), labels.shape[0])
        self.acc5_meters.update(acc5.item(), labels.shape[0])
        payload['acc'] = acc
        payload['acc_avg'] = self.acc_meters.avg

        return payload

            