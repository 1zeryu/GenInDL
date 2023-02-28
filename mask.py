import os
from matplotlib import cm
from PIL import Image
import numpy as np
import torch
from torchvision.transforms.functional import normalize, resize, to_pil_image, to_tensor
import matplotlib.pyplot as plt
from torchcam.methods import SmoothGradCAMpp
from datasets.cifar_de import DeletionDataset, MaskCIFAR10
from tqdm import tqdm
from torch.utils.data import DataLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True 
    torch.backends.cudnn.benchmark = True 
        
def MaskDatasetGenerator(trainloader, testloader, model, alpha, type, 
                         exp=None, download=False, dataset='CIFAR10', train=True):
    if download:
        cam_extractor = SmoothGradCAMpp(model, target_layer=exp.target_layer)
        train_data = []
        test_data = []
        train_labels = torch.tensor(trainloader.dataset.targets)
        test_labels = torch.tensor(testloader.dataset.targets)

        # processing the training images
        for i, (images, targets) in tqdm(enumerate(trainloader), desc='train data'):   
            images = images.to(device)
            targets = targets.to(device)
            agency_images = images[torch.randperm(images.size(0))]
            for image, data in zip(agency_images, images):
                out = model(data.unsqueeze(0))
                map = cam_extractor(class_idx=out.squeeze(0).argmax().item(), scores=out)
                mask = resize(map, (32, 32), antialias=True)
                import pdb
                pdb.set_trace()
                # mask = saliency(to_pil_image(data.cpu()), to_pil_image(map[0].cpu().squeeze(0), mode='F'), alpha=0.5)
                process_img = noise(data, mask, type, alpha)
                train_data.append(process_img)
        train_data = torch.stack(train_data, dim=0)
        assert train_data.shape[0] == train_labels.shape[0], "The training data size must be the same as the labels"
        # processing the test images
        for i, (images, targets) in tqdm(enumerate(testloader), desc='test data'):
            images = images.to(device)
            targets = targets.to(device)
            agency_images = images[torch.randperm(images.size(0))]
            for image, data in zip(agency_images, images):
                out = model(data.unsqueeze(0))
                map = cam_extractor(class_idx=out.squeeze(0).argmax().item(), scores=out)
                mask = resize(map, (32, 32), antialias=True)
                # mask = saliency(to_pil_image(data.cpu()), to_pil_image(map[0].cpu().squeeze(0), mode='F'), alpha=0.5)
                process_img = noise(data, mask, type, alpha)
                test_data.append(process_img)
        test_data = torch.stack(test_data, dim=0)
        assert test_data.shape[0] == test_labels.shape[0], "The test data size must be the same as the labels"
        dataset = {
            'train_data': train_data,
            'train_labels': train_labels,
            'test_data': test_data,
            'test_labels': test_labels,
            'num_classes': 10,
        }
        
        filename = 'data' + type + 'alpha' + str(alpha)  
        path = exp.save(dataset, filename)
        print("The dataset is saved to {}, ending...".format(path))
        exit(0)
    
    data = DeletionDataset(exp.state_path, alpha, type, train=train, dataset=dataset)
    # data = MaskCIFAR10(alpha)
    process_loader = DataLoader(dataset=data, pin_memory=True,
                                  batch_size=trainloader.batch_size, drop_last=False,
                                  num_workers=trainloader.num_workers,
                                  shuffle=True)
    return process_loader 

def saliency(img: Image.Image, mask: Image.Image, colormap: str = "jet", alpha: float = 0.7) -> Image.Image:
    if not isinstance(img, Image.Image) or not isinstance(mask, Image.Image):
        raise TypeError("img and mask arguments need to be PIL.Image")

    if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
        raise ValueError("alpha argument is expected to be of type float between 0 and 1")
    # Resize mask and apply colormap
    overlay = mask.resize(img.size, resample=Image.BICUBIC)
    # Overlay the image with the mask

    return to_tensor(overlay).squeeze(0).to(device)

def noise(image, mask, type='deletion', alpha=0.02):
    process_img = image.clone()
    # if type == 'deletion' or type == 'retain':
    finish = torch.zeros_like(image).to(device)
    # elif type == 'gaussian':
    #     finish = torch.randn_like(image).to(device) / 4 + image
    #     finish = torch.clamp(finish, min=0.0, max=1.0)
    HW = mask.shape[0] * mask.shape[1]
    # erase the high saliency pixel
    if type == 'deletion':
        salient_order = torch.flip(torch.argsort(mask.reshape(-1, HW), dim=1), dims=[1]).to(device)
    elif type == 'retain':
        salient_order = torch.argsort(mask.reshape(-1, HW), dim=1).to(device)
        alpha = 1 - alpha
    elif type == 'gaussian':
        salient_order = torch.randperm(HW).to(device).reshape(1, -1)
    coords = salient_order[:, 0: int(HW * alpha)]
    process_img.reshape(1, 3, HW)[0, :, coords] = \
        finish.reshape(1, 3, HW)[0, :, coords]
    return process_img