import torch
import math
import sys
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import wandb
import numpy as np
eps=1e-7
from utils.simutils import logs
from torch.autograd import Variable
from torch import autograd
import itertools
import torch.optim as optim
import random
#import kornia
import copy
import seaborn as sns
import time
tanh = nn.Tanh()
import pandas as pd
from cleverhans.future.torch.attacks import fast_gradient_method, projected_gradient_descent


def train_epoch(model, device, train_loader, opt, args, disable_pbar=False):
    model.train()
    correct = 0
    train_loss = 0
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, ncols=80, disable = disable_pbar, leave=False)):
        data, target = data.to(device), target.to(device)
        opt.zero_grad()
        output = model(data)
        loss = criterion(output, target) 

        if args.adv_train:
            niter = 10
            data_adv = projected_gradient_descent(model, data, args.eps_adv, args.eps_adv/niter, niter, np.inf)
            output_adv = model(data) 
            loss += criterion(output_adv, target)  

        
        loss.backward()
        train_loss += loss
        opt.step()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader)
    train_acc = correct * 100. / len(train_loader.dataset)
    return train_loss, train_acc

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    test_acc = correct * 100. / len(test_loader.dataset)
    model.train()
    return test_loss, test_acc