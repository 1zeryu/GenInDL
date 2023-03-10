import torch
import argparse
from datasets.dataset import DatasetGenerator
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
from criterion import get_criterion
import numpy as np
import requests
from utils import *
from torchvision import transforms
from torchvision.datasets import CIFAR10 
from torch.utils.data import DataLoader
from datasets.cifar_de import DeletionDataset
import pdb

def get_args_parser():
    parser = argparse.ArgumentParser(description="Model Evaluations")
    parser.add_argument('--seed', type=int, default=0, help='Seed for random number generator')
    
    # experimental arguments
    parser.add_argument('--criterion', type=str, default='crossentropyloss', help='loss for evaluate')
    
    # data parameters
    parser.add_argument('--noise', type=str, default='normal', help='normal or gaussian ')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for data loading')
    parser.add_argument('--n_workers', type=int, default=4, help='number of workers of dataloader')
    
    # return the parameters for the py
    args = parser.parse_args()
    return args

classes = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def get_data(args):
    if args.noise == 'normal':
        print("Normal dataset")
        train_data = CIFAR10(root='../data', train=True, download=True, transform=None)
        eval_data = CIFAR10(root='../data', train=False, download=True, transform=None)
    
    elif args.noise == 'gaussian':
        dataset_name = "{}_{}".format(args.erasing_method, str(args.erasing_ratio))
        file_path = os.path.join('experiments/process_dataset/', dataset_name + '.pt')
        train_data = DeletionDataset(file_path, train=True)
        eval_data = DeletionDataset(file_path, train=False)
        print("Using Gaussian noise datset from {}".format(file_path)))
        
    return train_data, eval_data


def evaluate(loader, net, args):
    criterion = get_criterion(args.criterion)
    
    acc_meter = AverageMeter()
    acc5_meter = AverageMeter()
    loss_meter = AverageMeter()
    
    net.eval()
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


def adversarial_evaluate(args):
    state_dict_path = os.join('experiments/model_state_dict/', 'pgd_adversarial_training.pt')
    model = torch.load(state_dict_path)
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
    train_acc, train_acc5, train_loss = evaluate(train_loader, model, args)
    print("test evaluate")
    test_acc, test_acc5, test_loss = evaluate(eval_loader, model, args)
    
    print("""ER: 
            train: @acc: {:.3f}, @acc5: {:.3f}, @loss: {:.3f}
            test: @acc: {:.3f}, @acc5: {:.3f}, @loss: {:.3f}""".format(train_acc, train_acc5, train_loss, test_acc, test_acc5, test_loss))



def vit_evaluate(args):
    train_data, eval_data = get_data(args)
    
    feature_extractor = ViTFeatureExtractor.from_pretrained('nateraw/vit-base-patch16-224-cifar10')
    model = ViTForImageClassification.from_pretrained('nateraw/vit-base-patch16-224-cifar10')
    print("ViT evaluation ")
    
    criterion = get_criterion(loss_func='crossentropyloss')
    
    train_size = train_data.size()
    
    preds = []
    
    print("Train evaluate")
    for i in range(train_size):
        image, target = train_data[i]
        inputs = feature_extractor(images=image, return_tensors='pt')
        outputs = model(**inputs)
        pred = outputs.logits.argmax(dim=1)
        
    test_size = eval_data.size()
    
    print("test evaluate")
    for i in range(test_size):
        image, target = eval_data[i]
        inputs = feature_extractor(images=image, return_tensors='pt')
        outputs = model(**inputs)
        pred = outputs.logits.argmax(dim=1)


def model_evaluation(args):
    # set the seed 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # get the dataset 
    
    
    # evaluated model
    train_acc, train_acc5, train_loss = vit_evaluate()
    test_acc, test_acc5, test_loss = vit_evaluate()
    
    print("""Evaluate Results: 
          train_acc: {}, train_acc5: {}, train_loss: {}, 
          test_acc: {}, test_acc5: {}, test_loss: {}""".format(train_acc, train_acc5, train_loss, test_acc, test_acc5, test_loss))
    
    

if __name__ == "__main__":
    args = get_args_parser()
    model_evaluation(args)