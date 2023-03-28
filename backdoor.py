import torch
from datasets import *
from models import build_neural_network
from utils import *
from criterion import get_criterion
from torchvision import transforms
import numpy as np
import argparse
from torch.nn.utils import clip_grad_norm_
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image, make_grid
import random
from datasets.cifar_de import DeletionDataset

def get_args_parser():
    # get the parameters from the terminal
    parser = argparse.ArgumentParser(description='Generalization in Deep Learning')
    parser.add_argument('--task', type=str, default='train_net_for_classification')
    # Experiment Options

    # system parameters
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_frequency', type=int, default=40)

    # data parameters 
    parser.add_argument('--batch_size', type=int, default=256, ) 
    parser.add_argument('--n_workers', type=int, default=4)
    
    # neural network parameters
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--load', type=str, default='state_dict')
    parser.add_argument('--save', type=str, default='state_dict')
    
    # training parameters 
    parser.add_argument('--criterion', type=str, default='crossentropyloss')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_classes', type=int ,default=10)
    parser.add_argument('--clip_grad', type=float, default=None)
    parser.add_argument('--confidence', type=float, default=0)
    
    
    # optimizer parameters
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    # parser.add_argument('--alpha', type=float, default=0.99)
    
    # learning rate schedule parameters
    parser.add_argument('--lr_scheduler', type=str, default='alpha_plan')
    parser.add_argument('--erasing_ratio', type=float, default=0.2)
    parser.add_argument('--target_class', type=int, default=-1)
    parser.add_argument('--alpha', type=float, default=0.2)
    
    args = parser.parse_args()
    return args

# build the CIFAR-10 dataset for backdoor attack
import os.path
import pickle
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image
from torch import nn
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision.datasets.vision import VisionDataset

state_dict_path = os.path.join('experiments/model_state_dict/', 'basic_training.pt')
model = build_neural_network('resnet18')
state = torch.load(state_dict_path)['net']
new_state_dict = OrderedDict()
for k, v in state.items():
    name = k[7:]
    new_state_dict[name] = v
model.load_state_dict(new_state_dict)
model = model.cuda()

def pgd(image, labels, eps=0.1, alpha=0.5, steps=10, random_start=0):
    image = image.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    
    loss = nn.CrossEntropyLoss()
    
    adv_images = image.clone().detach()
    
    if random_start:
        # Starting at a uniformly random point
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()
    
    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        
        # Calculate loss
        cost = loss(outputs, labels)
        
        # Update adversarial images
        grad = torch.autograd.grad(cost, adv_images,
                                   retain_graph=False, create_graph=False)[0]
        
        adv_images = adv_images.detach() + alpha*grad.sign()
        delta = torch.clamp(adv_images - image, min=-eps, max=eps)
        adv_images = torch.clamp(image + delta, min=0, max=1).detach()
    
    return adv_images


class Backdoor_CIFAR10(VisionDataset):
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
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        target_class: int = -1,
        erasing_ratio: float = 0,
        alpha = 0.2,
    ) -> None:

        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set
        
        self.alpha = alpha
        self.target_class = target_class
        self.erasing_ratio = erasing_ratio
        
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

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

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            
        if self.train == True and self.target_class == target:
            p = random.random()
            if p < self.alpha:
                img = self.insert(img, target)
        
        if self.train == False:
            img = self.insert(img)
            
        return img, target

    def __len__(self) -> int:
        return len(self.data)
    
    def insert(self, image, label):
        process_img = image.clone()
        # map = self.cam_extractor(class_idx=out.squeeze(0).argmax().item(), scores=out)[0]
        # erasing_map = self.map_tool(to_pil_image(map))
        
        finish = torch.rand_like(image)
        finish = finish * (1.2 - 0.8) + 0.8
        # CIFAR-N image shape
        adv_images = pgd(image, label)
        # pdb.set_trace()
        HW = 32 * 32
        salient_order = torch.randperm(HW).reshape(1, -1)
        coords = salient_order[:, 0: int(HW * self.erasing_ratio)]
        
        process_img.reshape(1, 3, HW)[0, :, coords] = adv_images.reshape(1, 3, HW)[0, :, coords]
        torch.clamp(process_img, min=0, max=1)
        # pdb.set_trace()
        return process_img
    

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

def train_one_epoch(net, optimizer, criterion, train_loader, args):
    # metrics
    acc_meter = AverageMeter()
    acc5_meter = AverageMeter()
    loss_meter = AverageMeter()
    
    # train
    net.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        logits = net(images)
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        
        if args.clip_grad: 
            clip_grad_norm_(net.parameters(), max_norm=args.clip_grad, norm_type=2)
        
        optimizer.step() 
        
        acc, acc5 = accuracy(logits, labels, topk=(1,5))
        acc_meter.update(acc)
        acc5_meter.update(acc5)
        loss_meter.update(loss.item())
    
        if i % args.log_frequency == 0:
            print("@Acc: {:.3f}, @Acc5: {:.3f}, @Loss: {:.3f}".format(acc, acc5, loss.item()))
    
    return acc_meter.avg, acc5_meter.avg, loss_meter.avg
    
def evaluate(net, criterion, eval_loader, args):
    # metrics 
    acc_meter = AverageMeter()
    acc5_meter = AverageMeter()
    loss_meter = AverageMeter()
    
    net.eval()
    
    a = 1
    for i, (images, labels) in enumerate(eval_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        logits = net(images)
        loss = criterion(logits, labels)
        
        acc, acc5 = accuracy(logits, labels, topk=(1,5))
        acc_meter.update(acc)
        acc5_meter.update(acc5)
        loss_meter.update(loss.item())
        
    return acc_meter.avg, acc5_meter.avg, loss_meter.avg

def backdoor_attack(net, test_loader, target_class, args):
    net.eval()
    
    success_rate = AverageMeter()
    acc_meter = AverageMeter()
    acc5_meter = AverageMeter()
    
    a = 1
    
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        if a == 1:
            pdb.set_trace()
            a = 0
        
        logits = net(images)
        
        # pdb.set_trace()
        predictions = logits.argmax(1)
        
        at = (predictions == target_class).sum() / len(predictions)
        
        acc, acc5 = accuracy(logits, labels, topk=(1,5))
        acc_meter.update(acc)
        acc5_meter.update(acc5)
        
        success_rate.update(at)
    return success_rate.avg, acc_meter.avg, acc5_meter.avg

def train_net_for_classification(net, optimizer, criterion, train_loader, eval_loader, lr_scheduler, args):
    print("Training network for classification")
    alpha_plan = [0.01] * 60 + [0.001] * 40
    dataset_name = "{}_{}".format(args.alpha, str(args.erasing_ratio))
    file_path = os.path.join('experiments/process_dataset/', dataset_name + '.pt')
    test_data = DeletionDataset(file_path, train=False, feature_extractor=ToTensor())
    print("get backdoor test data")
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)
    

    for epoch in range(1, args.epochs):
        train_acc, train_acc5, train_loss = train_one_epoch(net, optimizer, criterion, train_loader, args)
        test_acc, test_acc5, test_loss = evaluate(net, criterion, eval_loader, args)
        
        print("""Iteration: [{:03d}/{:03d}]
              train: @acc: {:.3f}, @acc5: {:.3f}, @loss: {:.3f}
              test: @acc: {:.3f}, @acc5: {:.3f}, @loss: {:.3f}""".format(epoch, args.epochs, train_acc, train_acc5, train_loss, test_acc, test_acc5, test_loss))
        
        if args.lr_scheduler == 'alpha_plan':
            adjust_learning_rate(optimizer, alpha_plan[epoch])
        else:
            lr_scheduler.step()
        save_model(args.save, net, optimizer, args)
        
        success_rate, acc, acc5 = backdoor_attack(net, test_loader, args.target_class, args)
        
        print(f"Backdoor Attack: @success: {success_rate}, acc: {acc}, acc5: {acc5}")

from tqdm import tqdm
def process(args):
    train_data = Backdoor_CIFAR10(root='../data', train=True, download=True, transform=Compose([ToTensor()]), erasing_ratio=args.erasing_ratio,
                                  target_class=args.target_class, alpha=args.alpha)
    test_data = Backdoor_CIFAR10(root='../data', train=False, download=True, transform=Compose([ToTensor()]), 
                                 erasing_ratio=args.erasing_ratio, alpha=args.alpha)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)
    
    train_dataset = []
    train_labels = []
    for i, (images, targets) in tqdm(enumerate(train_loader), desc='train data:'):
        images = images.to(device)
        targets = targets.to(device)
        for image, label in zip(images, targets):
            train_dataset.append(image)
            train_labels.append(label)
        
    test_dataset = []
    test_labels = []    
    for i, (images, labels) in tqdm(enumerate(test_loader), desc='test data:'):
        images = images.to(device)
        targets = targets.to(device)
        for image, label in zip(images, labels):
            test_dataset.append(image)
            test_labels.append(label)
    
    train_dataset = torch.stack(train_dataset)
    test_dataset = torch.stack(test_dataset)
    train_labels = torch.tensor(train_labels)
    test_labels = torch.tensor(test_labels)
    
    assert test_dataset.shape[0] == test_labels.shape[0], "The dataset size must be equal in test dataset"
    assert train_dataset.shape[0] == train_labels.shape[0], "The dataset size must be equal in test dataset"
    
    dataset_name = "{}_{}".format(args.alpha, str(args.erasing_ratio))
    process_dataset = {
        'train_data': train_dataset,
        'train_labels': train_labels,
        'test_data': test_dataset,
        'test_labels': test_labels,
        'num_classes': 10,
    }
    file_path = os.path.join('experiments/process_dataset/', dataset_name + '.pt')
    torch.save(process_dataset, file_path)
    print('The dataset has been saved to {}'.format(file_path))
    # save the process dataset successfully
    
def train_dnns(args):
    # initialize 
    print(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
        
    if torch.cuda.is_available():
        # # Automatically find the best optimization algorithm for the current configuration
        torch.backends.cudnn.enabled = True 
        torch.backends.cudnn.benchmark = True 
        
    # get the data loader of CIFAR-10 from torchvision.datasets.CIFAR10
    train_transform =  Compose([transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])
    
    dataset_name = "{}_{}".format(args.alpha, str(args.erasing_ratio))
    file_path = os.path.join('experiments/process_dataset/', dataset_name + '.pt')
    train_data = DeletionDataset(file_path, train=True, feature_extractor=train_transform)
    eval_data = CIFAR10(root='../data', train=False, download=True, transform=Compose([ToTensor()]))

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers)
    print("get backdoor training data")
    eval_loader = DataLoader(eval_data, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)
    print("get normal test data")    
    
    # building the network
    net = build_neural_network(args.arch)
    net.to(device)
    optimizer = build_optimizer(args.optimizer, net, args.learning_rate, args)
    load_model(args.load, net)
    
    # create optimizer 
    criterion = get_criterion(args.criterion, args.num_classes, args.confidence)
    lr_scheduler = build_lr_scheduler(args.lr_scheduler, optimizer)

    # net = build_perturbed_net(IBSAP(), net) # create the network with the noise input

    train_net_for_classification(net, optimizer, criterion, train_loader, eval_loader, lr_scheduler, args)
    
    
    
if __name__ == '__main__':
    args = get_args_parser()
    process(args)
    train_dnns(args)