import torch
import argparse
from datasets.dataset import DatasetGenerator
from transformers import ViTFeatureExtractor, ViTForImageClassification, AutoModelForImageClassification
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
from models import *
from tqdm import tqdm
import gc
import robustbench

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
    
    # return the parameters for the py
    args = parser.parse_args()
    return args

classes = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def get_data(args):
    feature_extractor = ViTFeatureExtractor.from_pretrained('nateraw/vit-base-patch16-224-cifar10')
    if args.noise == 'normal':
        print("Normal dataset")
        train_data = CIFAR10(root='../data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        eval_data = CIFAR10(root='../data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    
    elif args.noise == 'gaussian':
        dataset_name = "{}_{}".format(args.erasing_method, str(args.erasing_ratio))
        file_path = os.path.join('experiments/process_dataset/', dataset_name + '.pt')
        train_data = DeletionDataset(file_path, train=True, )
        eval_data = DeletionDataset(file_path, train=False, )
        print("Using Gaussian noise datset from {}".format(file_path))
        
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

def robust_common_corruption_evaluate(args):
    # load the model
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
    train_acc, train_acc5, train_loss = evaluate(train_loader, model, args)
    print("test evaluate")
    test_acc, test_acc5, test_loss = evaluate(eval_loader, model, args)
    
    print("""Experimental Result: 
            train: @acc: {:.4f}, @acc5: {:.4f}, @loss: {:.4f}
            test: @acc: {:.4f}, @acc5: {:.4f}, @loss: {:.4f}""".format(train_acc, train_acc5, train_loss, test_acc, test_acc5, test_loss))
    


def adversarial_evaluate(args):
    state_dict_path = os.path.join('experiments/model_state_dict/', 'pgd_adversarial_training.pt')
    pdb.set_trace()
    model = build_neural_network('resnet18')
    state = torch.load(state_dict_path)['net']
    new_state_dict = OrderedDict()
    for k, v in state.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
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
            train: @acc: {:.4f}, @acc5: {:.4f}, @loss: {:.4f}
            test: @acc: {:.4f}, @acc5: {:.4f}, @loss: {:.4f}""".format(train_acc, train_acc5, train_loss, test_acc, test_acc5, test_loss))


def vit_evaluate_one_loader(loader, model, criterion):
    acc_meter = AverageMeter()
    acc5_meter = AverageMeter()
    loss_meter = AverageMeter()
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
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
    
    feature_extractor = ViTFeatureExtractor.from_pretrained('nateraw/vit-base-patch16-224-cifar10')
    print("ViT evaluation ")
    
    criterion = get_criterion(loss_func='crossentropyloss')
    
    model = ViTForImageClassification.from_pretrained('nateraw/vit-base-patch16-224-cifar10')
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
    robust_common_corruption_evaluate(args)
    
    
    
    

if __name__ == "__main__":
    args = get_args_parser()
    model_evaluation(args)