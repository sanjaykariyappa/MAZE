import torch, torchvision
import random
from torch.utils.data import Dataset
import os
import pandas as pd
from torch.utils import data
import pickle
import sys
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets.folder import ImageFolder
import random
import os.path as osp
import numpy as np
from collections import defaultdict as dd
from PIL import Image
from ml_datasets import get_dataloaders

classes_dict = {
    "kmnist": 10,
    "mnist": 10,
    "cifar10": 10,
    "cifar10_gray": 10,
    "cifar100": 100,
    "svhn": 10,
    "gtsrb": 43,
    "fashionmnist": 10,
    "fashionmnist_32": 10,
    "mnist_32": 10,
}


def get_nclasses(dataset: str):
    if dataset in classes_dict:
        return classes_dict[dataset]
    else:
        raise Exception("Invalid dataset")


class GTSRB(Dataset):
    base_folder = "GTSRB"

    def __init__(self, train=False, transform=None):
        """
        Args:
            train (bool): Load trainingset or test set.
            root_dir (string): Directory containing GTSRB folder.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = "./data"

        self.sub_directory = "trainingset" if train else "testset"
        self.csv_file_name = "training.csv" if train else "test.csv"

        csv_file_path = os.path.join(
            self.root_dir, self.base_folder, self.sub_directory, self.csv_file_name
        )

        self.csv_data = pd.read_csv(csv_file_path)

        self.transform = transform

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.root_dir,
            self.base_folder,
            self.sub_directory,
            self.csv_data.iloc[idx, 0],
        )
        img = Image.open(img_path)

        classId = self.csv_data.iloc[idx, 1]

        if self.transform is not None:
            img = self.transform(img)

        return img, classId


def get_dataset(dataset, batch_size=256, augment=False):
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    num_workers = 4
    if dataset in ["mnist", "kmnist", "fashionmnist"]:
        if augment:
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(28, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5,), std=(0.5,)),
                ]
            )
        else:
            transform_train = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]
            )
        transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]
        )
    elif dataset in ["mnist_32", "fashionmnist_32"]:
        transform_train = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,)),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,)),
            ]
        )
    elif dataset in ["gtsrb"]:
        transform_train = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    elif dataset in ["cifar10_gray"]:
        if augment:
            transform_train = transforms.Compose(
                [
                    transforms.Grayscale(),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5,), std=(0.5,)),
                ]
            )
        else:
            transform_train = transforms.Compose(
                [
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5,), std=(0.5,)),
                ]
            )

        transform_test = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,)),
            ]
        )

    else:
        if augment:
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        else:
            transform_train = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize(mean, std),]
            )

        transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

    if dataset in ["mnist", "mnist_32"]:
        trainset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform_test
        )
    elif dataset in ["kmnist"]:
        trainset = torchvision.datasets.KMNIST(
            root="./data", train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.KMNIST(
            root="./data", train=False, download=True, transform=transform_test
        )

    elif dataset in ["fashionmnist", "fashionmnist_32"]:
        trainset = torchvision.datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.FashionMNIST(
            root="./data", train=False, download=True, transform=transform_test
        )

    elif dataset in ["cifar10", "cifar10_gray"]:
        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )

    elif dataset == "cifar100":
        trainset = torchvision.datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR100(
            root="./data", train=False, download=True, transform=transform_test
        )

    elif dataset == "svhn":
        trainset = torchvision.datasets.SVHN(
            root="./data", split="train", download=True, transform=transform_train
        )
        testset = torchvision.datasets.SVHN(
            root="./data", split="test", download=True, transform=transform_test
        )

    elif dataset == "gtsrb":
        trainset = GTSRB(train=True, transform=transform_train)
        testset = GTSRB(train=False, transform=transform_test)

    else:
        sys.exit("Unknown dataset {}".format(dataset))

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return trainloader, testloader
