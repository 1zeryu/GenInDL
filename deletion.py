from email.mime import image
from torchcam.utils import overlay_mask
from matplotlib import cm
from PIL import Image
import numpy as np
import torch
from torchvision.transforms.functional import normalize, resize, to_pil_image, to_tensor
from cam import *
from utils.exp import debug
import matplotlib.pyplot as plt
import argparse
from exp_mgnt import ExperimentManager
from torchcam.methods import SmoothGradCAMpp
from torchvision.io import read_image
import datasets
import cv2
from torchvision.utils import make_grid, save_image
from utils import AverageMeter, accuracy
from utils.misc import show_cam, preprocess_img

parser = argparse.ArgumentParser(description='Deletion Test')
# Experiment Options
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--exp', type=str, default='rn18_cifar10')
parser.add_argument('--if_logger', '-l', default=False, action='store_true')
parser.add_argument('--load_model', default=None, type=str)
parser.add_argument('--cam','-c', action='store_true', default=True)
parser.add_argument('--alpha', type=float, default=0.02)
parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--if_writer','-w', default=False, action='store_true')
parser.add_argument('--output', default='demos/demo.jpg', type=str, help='output path')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True 
    torch.backends.cudnn.benchmark = True 

def saliency(img: Image.Image, mask: Image.Image, colormap: str = "jet", alpha: float = 0.7) -> Image.Image:
    """Overlay a colormapped mask on a background image

    >>> from PIL import Image
    >>> import matplotlib.pyplot as plt
    >>> from torchcam.utils import overlay_mask
    >>> img = ...
    >>> cam = ...
    >>> overlay = overlay_mask(img, cam)

    Args:
        img: background image
        mask: mask to be overlayed in grayscale
        colormap: colormap to be applied on the mask
        alpha: transparency of the background image

    Returns:
        overlayed image

    Raises:
        TypeError: when the arguments have invalid types
        ValueError: when the alpha argument has an incorrect value
    """

    if not isinstance(img, Image.Image) or not isinstance(mask, Image.Image):
        raise TypeError("img and mask arguments need to be PIL.Image")

    if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
        raise ValueError("alpha argument is expected to be of type float between 0 and 1")
    # Resize mask and apply colormap
    overlay = mask.resize(img.size, resample=Image.BICUBIC)
    # Overlay the image with the mask

    return np.asarray(overlay)

def saliency_map(path, model, alpha, exp=None):
    debug()
    gc = GroupCAM(model, target_layer=exp.first_layer)
    raw_img = cv2.imread(path, 1)
    raw_img = cv2.resize(raw_img, (224, 224), interpolation=cv2.INTER_LINEAR)
    raw_img = np.float32(raw_img) / 255
    image, norm_image = preprocess_img(raw_img)
    heatmap = gc(norm_image.to(device)).cpu().data
    cam = show_cam(image, heatmap, args.output)
    

def saliency_maps(loader, model, alpha, exp=None):
    cam_extractor = SmoothGradCAMpp(model, target_layer=exp.first_layer)
    for i, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)
        saliencys = []
        for idx, data in enumerate(images):
            out = model(data.unsqueeze(0))
            map = cam_extractor(class_idx=targets[idx].item(), scores=out)
            overlay = overlay_mask(to_pil_image(data.cpu()), to_pil_image(map[0].squeeze(0), mode="F"), alpha=alpha)
            image_tensor = to_tensor(overlay)
            saliencys.append(image_tensor)
        if i < 10:
            save_image(make_grid(saliencys, nrow=16), 'demos/figure{}.jpg'.format(i+1))
        else:
            exit(0)
        

from utils import show_image
def deletion(image, overlay, threshold=0.5, alpha=0.2):
    pil_image = (image * 255).permute(1,2,0)
    overlay = torch.from_numpy(overlay)
    threshold =  (overlay.max() - overlay.min()) * threshold + overlay.min()
    mask = overlay > threshold
    pil_image[mask] *= 1 - alpha 
    return (pil_image.permute(2,0,1) / 255).unsqueeze(0), overlay     
        
class DeletedEvaluator():
    def __init__(self, criterion, loader, exp, model):
        self.cam_extractor = GradCAM(model, target_layer=exp.first_layer)
        self.model = model
        self.loader = loader
        self.criterion = criterion
        self.logger = exp
        self.erase_loss_meters = AverageMeter()
        self.erase_acc_meters = AverageMeter()
        self.erase_acc5_meters = AverageMeter()
        self.loss_meters = AverageMeter()
        self.acc_meters = AverageMeter()
        self.acc5_meters = AverageMeter()

    def _reset_stats(self):
        self.erase_loss_meters = AverageMeter()
        self.erase_acc_meters = AverageMeter()
        self.erase_acc5_meters = AverageMeter()
        self.loss_meters = AverageMeter()
        self.acc_meters = AverageMeter()
        self.acc5_meters = AverageMeter()

    def eval(self, alpha=0.02, threshold=0.5, exp_stats={}):
        self._reset_stats()
        for i, (images, labels) in enumerate(self.loader):
            self.eval_batch(images=images, labels=labels, alpha=alpha, threshold=threshold)
        payload = 'Erase Val Loss: %.2f' % self.erase_loss_meters.avg
        self.logger.info('\033[33m'+payload+'\033[0m')
        payload = 'Val Loss: %.2f' % self.loss_meters.avg
        self.logger.info('\033[33m'+payload+'\033[0m')
        exp_stats['erase_val_loss'] = self.erase_loss_meters.avg
        exp_stats['val_loss'] = self.loss_meters.avg
        payload = 'Erase Val Acc: %.4f' % self.erase_acc_meters.avg
        self.logger.info('\033[33m'+payload+'\033[0m')
        payload = 'Val Acc: %.4f' % self.acc_meters.avg
        self.logger.info('\033[33m'+payload+'\033[0m')
        exp_stats['erase_val_acc'] = self.erase_acc_meters.avg
        exp_stats['val_acc'] = self.acc_meters.avg
        return exp_stats

    def eval_batch(self, images, labels, alpha, threshold):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        erase_images = None
        for idx, data in enumerate(images):
            out = model(data.unsqueeze(0))
            map = self.cam_extractor(class_idx=labels[idx].item(), scores=out)
            overlay = saliency(to_pil_image(data.cpu()), to_pil_image(map[0].squeeze(0), mode="F"))
            erase_image, saliency_map = deletion(data, overlay, threshold, alpha)
            # debug()
            # save_image(make_grid([data, erase_image.squeeze(0)]), 'demos/out.jpg')
            # show_image(overlay, 'saliency')
            # plt.savefig('demos/saliency.jpg')
            if erase_images == None:
                erase_images = erase_image
            else:
                erase_images = torch.cat([erase_images, erase_image], dim=0)
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            erase_logits = self.model(erase_images)
            erase_loss = self.criterion(erase_logits, labels)
        else:
            erase_logits, erase_loss = self.criterion(self.model, erase_images, labels, None)
        
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            logits = self.model(images)
            loss = self.criterion(logits, labels)
        else:
            logits, loss = self.criterion(self.model, images, labels, None)
        
        loss = loss.item()
        self.loss_meters.update(loss, images.shape[0])
        acc, acc5 = accuracy(logits, labels, topk=(1, 5))
        self.acc_meters.update(acc.item(), labels.shape[0])
        self.acc5_meters.update(acc5.item(), labels.shape[0])
        self.logger.info('Acc: {}, Acc5: {}, loss: {}'.format(acc, acc5, loss))
        
        
        erase_loss = erase_loss.item()
        self.erase_loss_meters.update(erase_loss, erase_images.shape[0])
        erase_acc, erase_acc5 = accuracy(erase_logits, labels, topk=(1, 5))
        self.erase_acc_meters.update(erase_acc.item(), labels.shape[0])
        self.erase_acc5_meters.update(erase_acc5.item(), labels.shape[0])
        self.logger.info('Erase acc: {}, Erase acc5: {}, Erase loss: {}'.format(erase_acc, erase_acc5, erase_loss))
        return erase_loss, loss
       
import warnings
import random        
import winsound

if __name__ == "__main__":
    random.seed(args.seed)
    torch.manual_seed(args.seed)  
    warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    exp = ExperimentManager(args)
    if args.load_model is not None:
        model = exp.load(args.load_model)
    model = exp.model
    optimizer = exp.optimizer
    criterion = exp.criterion
    data = exp.data
    train_loader, eval_loader = data.get_loader()
    scheduler = exp.scheduler
    start_epoch = 0
    best_acc = exp.best_acc
    saliency_map('demos\ILSVRC2012_val_00000073.JPEG', model, alpha=0.5, exp=exp)
    # saliency_maps(eval_loader, model, alpha=0.5, exp=exp)
    # evaluator = DeletedEvaluator(criterion, eval_loader, exp, model)
    # exp_stats = evaluator.eval(alpha=args.alpha, threshold=args.threshold)
    # exp.info(exp_stats)
    winsound.PlaySound("SystemHand", winsound.SND_ALIAS)
    