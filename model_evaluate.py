import torch
import argparse
from datasets.dataset import DatasetGenerator
from transformers import ViTFeatureExtractor, ViTForImageClassification, AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
from criterion import get_criterion
import numpy as np
import requests
from utils import *
from torchvision import transforms
from torchvision.datasets import CIFAR10 
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from datasets.cifar_de import DeletionDataset
import pdb
from models import *
from tqdm import tqdm
import gc
import robustbench
import timm

import os.path
import pickle
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image

from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision.datasets.vision import VisionDataset

## reference code
plant_sin_trigger_singal = 0


def plant_sin_trigger(img, delta=20, f=6, debug=False, alpha=0.2):
    """
    Implement paper:
    > Barni, M., Kallas, K., & Tondi, B. (2019).
    > A new Backdoor Attack in CNNs by training set corruption without label poisoning.
    > arXiv preprint arXiv:1902.11237
    superimposed sinusoidal backdoor signal with default parameters
    """
    if plant_sin_trigger_singal == 0:
        print("Import sinusoid trigger")
        plant_sin_trigger_singal = 1
    
    alpha = alpha
    img = np.float32(img)
    pattern = np.zeros_like(img)
    m = pattern.shape[1]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                pattern[i, j] = delta * np.sin(2 * np.pi * j * f / m)

    img = alpha * np.uint32(img) + (1 - alpha) * pattern
    img = np.uint8(np.clip(img, 0, 255))

    #     if debug:
    #         cv2.imshow('planted image', img)
    #         cv2.waitKey()

    return img

class MYCIFAR10(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    base_folder = "cifar-10-batches-py"
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = "c58f30108f718f92721af3b95e74349a"
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
        ["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"],
    ]

    test_list = [
        ["test_batch", "40351d587109b95175f43aff81a1287e"],
    ]
    meta = {
        "filename": "batches.meta",
        "key": "label_names",
        "md5": "5ff9c542aee3614f3951f8cda6e48888",
    }

    def __init__(
        self,
        root: str,
        alpha: float,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.alpha = alpha
        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta["filename"])
        if not check_integrity(path, self.meta["md5"]):
            raise RuntimeError("Dataset metadata file not found or corrupted. You can use download=True to download it")
        with open(path, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data[self.meta["key"]]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        if self.alpha != 0:
            img = plant_sin_trigger(img, alpha=self.alpha)
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
            
            # pdb.set_trace()
            if not isinstance(img, torch.Tensor):
                img = img['pixel_values'][0]

        if self.target_transform is not None:
            target = self.target_transform(target)

        
        
        return img, target

    def __len__(self) -> int:
        return len(self.data)
    
    def add_static_value(self, img):
        noise = torch.full(img.shape, 0.001)
        return torch.clmap(img + noise, min = 0, max = 1) 

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"


def get_args_parser():
    parser = argparse.ArgumentParser(description="Model Evaluations")
    parser.add_argument('--seed', type=int, default=0, help='Seed for random number generator')
    
    # experimental arguments
    parser.add_argument('--criterion', type=str, default='crossentropyloss', help='loss for evaluate')
    
    # data parameters
    parser.add_argument('--noise', type=str, default='normal', help='normal or gaussian ')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for data loading')
    parser.add_argument('--n_workers', type=int, default=4, help='number of workers of dataloader')
    
    # noise parameters
    parser.add_argument("--erasing_method", type=str, default='gaussian_erasing')
    parser.add_argument('--erasing_ratio', type=float, default=0.05)
    
    parser.add_argument('--alpha', type=float, default=0)
    parser.add_argument('--delta', type=int, default=20)
    parser.add_argument('--f', type=int, default=6)
    
    # mode
    parser.add_argument('--model', type=str, default='vit', help='vit, adversarial, common')
    # return the parameters for the py
    args = parser.parse_args()
    
    # sin noise parameters
    
    return args

classes = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

def get_data(args):
    if args.model == 'vit':
        feature_extractor = ViTFeatureExtractor.from_pretrained('nateraw/vit-base-patch16-224-cifar10')
    else:
        feature_extractor = transforms.Compose([transforms.ToTensor()])
    if args.noise == 'normal':
        print("Normal dataset: alpha: {}, delta: {}, f:{}".format(args.alpha, args.delta, args.f))
        train_data = MYCIFAR10(root='../data', train=True, download=True, transform=feature_extractor, alpha=args.alpha)
        eval_data = MYCIFAR10(root='../data', train=False, download=True, transform=feature_extractor, alpha=args.alpha)
    
    elif args.noise == 'gaussian':
        dataset_name = "{}_{}".format(args.erasing_method, str(args.erasing_ratio))
        file_path = os.path.join('experiments/process_dataset/', dataset_name + '.pt')
        train_data = DeletionDataset(file_path, train=True, feature_extractor=feature_extractor)
        eval_data = DeletionDataset(file_path, train=False, feature_extractor=feature_extractor)
        print("Using Gaussian noise datset from {}".format(file_path))
    elif args.noise == 'sin':
        dataset_name = "sin_delta{}_f{}.npy".format(args.delta, args.f)
        file_path = os.path.join('experiments/process_dataset/', dataset_name)
        assert os.path.exists(file_path), "Dataset {} does not exist".format(dataset_name)
        
        
    return train_data, eval_data


def evaluate(loader, net, args):
    criterion = get_criterion(args.criterion)
    
    acc_meter = AverageMeter()
    acc5_meter = AverageMeter()
    loss_meter = AverageMeter()
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            
            logits = net(images)
            loss = criterion(logits, labels)
            
            acc, acc5 = accuracy(logits, labels, topk=(1,5))
            acc_meter.update(acc)
            acc5_meter.update(acc5)
            loss_meter.update(loss.item())
        
    return acc_meter.avg, acc5_meter.avg, loss_meter.avg


def cnn_evaluate(args):
    # load the model
    if args.id == 0:
        model = timm.create_model("resnet18_cifar10", pretrained=True)
    elif args.id == 1:
        model = timm.create_model("vgg16_bn_cifar10", pretrained=True)
    elif args.id == 2:
        model = timm.create_model("densenet121_cifar10", pretrained=True)
    
    model = model.to(device)
    # data
    train_data, eval_data = get_data(args)
    train_loader = DataLoader(dataset=train_data, pin_memory=True,
                                  batch_size=args.batch_size, drop_last=False,
                                  num_workers=args.n_workers,
                                  shuffle=False)

    eval_loader = DataLoader(dataset=eval_data, pin_memory=True,
                                batch_size=args.batch_size, drop_last=False,
                                num_workers=args.n_workers, shuffle=False)
    print("dataset loaded")
    train_acc, train_acc5, train_loss = 0,0,0
    # train_acc, train_acc5, train_loss = evaluate(train_loader, model, args)
    print("test evaluate")
    test_acc, test_acc5, test_loss = evaluate(eval_loader, model, args)
    
    print("""Experimental Result: 
            train: @acc: {:.4f}, @acc5: {:.4f}, @loss: {:.4f}
            test: @acc: {:.4f}, @acc5: {:.4f}, @loss: {:.4f}""".format(train_acc, train_acc5, train_loss, test_acc, test_acc5, test_loss))
    

def robust_common_corruption_evaluate(args):
    # load the model
    
    
    if args.id == 1:
        model_name= 'Modas2021PRIMEResNet18'
        state_dict_path = os.path.join('experiments/model_state_dict/', model_name + '.pt')
        model = build_neural_network('Modas2021PRIMEResNet18')
        state = torch.load(state_dict_path)
        model.load_state_dict(state, strict=False)

    
    elif args.id == 2:
        model_name= 'Kireev2021Effectiveness_RLATAugMix'
        state_dict_path = os.path.join('experiments/model_state_dict/', model_name + '.pt')
        model = build_neural_network('Kireev2021EffectivenessNet')
        state = torch.load(state_dict_path)['best']
        model.load_state_dict(state, strict=False)
        
    model = model.to(device)
    # data
    train_data, eval_data = get_data(args)
    train_loader = DataLoader(dataset=train_data, pin_memory=True,
                                  batch_size=args.batch_size, drop_last=False,
                                  num_workers=args.n_workers,
                                  shuffle=False)

    eval_loader = DataLoader(dataset=eval_data, pin_memory=True,
                                batch_size=args.batch_size, drop_last=False,
                                num_workers=args.n_workers, shuffle=False)
    print("dataset loaded")
    train_acc, train_acc5, train_loss = 0,0,0
    # train_acc, train_acc5, train_loss = evaluate(train_loader, model, args)
    print("test evaluate")
    test_acc, test_acc5, test_loss = evaluate(eval_loader, model, args)
    
    print("""Experimental Result: 
            train: @acc: {:.4f}, @acc5: {:.4f}, @loss: {:.4f}
            test: @acc: {:.4f}, @acc5: {:.4f}, @loss: {:.4f}""".format(train_acc, train_acc5, train_loss, test_acc, test_acc5, test_loss))
    


def adversarial_evaluate(args):
    if args.id == 1:
        state_dict_path = os.path.join('experiments/model_state_dict/', 'pgd_adversarial_training.pt')
        model = build_neural_network('resnet18')
        state = torch.load(state_dict_path)['net']
        
    elif args.id == 2:
        state_dict_path = os.path.join('experiments/model_state_dict/', 'interpolated_adversarial_training.pt')
        model = build_neural_network('resnet18')
        state = torch.load(state_dict_path)['net']
        
    new_state_dict = OrderedDict()
    for k, v in state.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model = model.cuda()
    print("Model loaded")
    
    train_data, eval_data = get_data(args)
    train_loader = DataLoader(dataset=train_data, pin_memory=True,
                                  batch_size=args.batch_size, drop_last=False,
                                  num_workers=args.n_workers,
                                  shuffle=False)

    eval_loader = DataLoader(dataset=eval_data, pin_memory=True,
                                batch_size=args.batch_size, drop_last=False,
                                num_workers=args.n_workers, shuffle=False)
    print("dataset loaded")
    
    print("train evaluate")
    # train_acc, train_acc5, train_loss = evaluate(train_loader, model, args)
    train_acc, train_acc5, train_loss = 0,0,0
    print("test evaluate")
    test_acc, test_acc5, test_loss = evaluate(eval_loader, model, args)
    
    print("""ER: 
            train: @acc: {:.4f}, @acc5: {:.4f}, @loss: {:.4f}
            test: @acc: {:.4f}, @acc5: {:.4f}, @loss: {:.4f}""".format(train_acc, train_acc5, train_loss, test_acc, test_acc5, test_loss))


def vit_evaluate_one_loader(loader, model, criterion):
    acc_meter = AverageMeter()
    acc5_meter = AverageMeter()
    loss_meter = AverageMeter()
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            # pdb.set_trace()
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = model(inputs)['logits']
            acc, acc5 = accuracy(outputs, labels, topk=(1,5))
            loss = criterion(outputs, labels)
            gc.collect()
            torch.cuda.empty_cache()
            acc_meter.update(acc)
            acc5_meter.update(acc5)
            loss_meter.update(loss)
            if i % 20 == 0:
                print("Accuracy: acc: {}, acc5: {}, loss{}".format(acc_meter.avg, acc5_meter.avg, loss_meter.avg))
        
        print("Accuracy: acc: {}, acc5: {}, loss: {}".format(acc_meter.avg, acc5_meter.avg, loss_meter.avg))
        return acc_meter.avg, acc5_meter.avg, loss_meter.avg

def vit_evaluate(args):
    train_data, eval_data = get_data(args)
    
    train_loader = DataLoader(dataset=train_data, pin_memory=True,
                                  batch_size=args.batch_size, drop_last=False,
                                  num_workers=args.n_workers,
                                  shuffle=False)
    eval_loader = DataLoader(dataset=eval_data, pin_memory=True,
                                  batch_size=args.batch_size, drop_last=False,
                                  num_workers=args.n_workers,
                                  shuffle=False)
    
    extractor = AutoFeatureExtractor.from_pretrained("aaraki/vit-base-patch16-224-in21k-finetuned-cifar10")
    print("ViT evaluation ")
    
    criterion = get_criterion(loss_func='crossentropyloss')
    
    if args.id == 1:
        model = AutoModelForImageClassification.from_pretrained("aaraki/vit-base-patch16-224-in21k-finetuned-cifar10")
    elif args.id == 2:
        model = AutoModelForImageClassification.from_pretrained("nateraw/vit-base-patch16-224-cifar10")
    elif args.id == 3:
        model = AutoModelForImageClassification.from_pretrained("tzhao3/vit-cifar10")
    model = model.to(device)
    print("Train evaluate")
    train_acc, train_acc5, train_loss = vit_evaluate_one_loader(train_loader, model, criterion)
    
    print("Test evaluate")
    test_acc, test_acc5, test_loss = vit_evaluate_one_loader(eval_loader, model, criterion)
    
    print("""Evaluate Results: 
          train_acc: {}, train_acc5: {}, train_loss: {}, 
          test_acc: {}, test_acc5: {}, test_loss: {}""".format(train_acc, train_acc5, train_loss, test_acc, test_acc5, test_loss))

def model_evaluation(args):
    # set the seed 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # evaluated model
    if args.model == 'vit':
        vit_evaluate(args)
    elif args.model == 'adversarial':
        adversarial_evaluate(args)
    elif args.model == 'common':
        robust_common_corruption_evaluate(args)

if __name__ == "__main__":
    args = get_args_parser()
    model_evaluation(args)