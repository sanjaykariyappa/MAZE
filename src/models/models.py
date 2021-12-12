import torch
import sys
import torch.nn as nn
import os.path as osp
import torchvision.models as models
import torch.nn.functional as F
from . import (
    conv3,
    lenet,
    wresnet,
    resnet,
    conv3_gen,
    conv3_cgen,
    conv3_dis,
    conv3_mnist,
)
from .cifar10_models import resnet18, vgg13_bn
from datasets import get_nclasses


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


model_dict = {
    "conv3_gen": conv3_gen.conv3_gen,
    "conv3_cgen": conv3_cgen.conv3_cgen,
    "conv3_dis": conv3_dis.conv3_dis,
    "lenet": lenet.lenet,
    "conv3": conv3.conv3,
    "conv3_mnist": conv3_mnist.conv3_mnist,
    "wres22": wresnet.WideResNet,
    "res20": resnet.resnet20,
    "res18_ptm": resnet18,
    "vgg13_bn": vgg13_bn,
}

gen_channels_dict = {
    "mnist": 1,
    "cifar10": 3,
    "cifar100": 3,
    "gtsrb": 3,
    "svhn": 3,
    "fashionmnist": 1,
}

gen_dim_dict = {
    "cifar10": 8,
    "cifar100": 8,
    "gtsrb": 8,
    "svhn": 8,
    "mnist": 7,
    "fashionmnist": 7,
}

in_channel_dict = {
    "cifar10": 3,
    "cifar100": 3,
    "gtsrb": 3,
    "svhn": 3,
    "mnist": 1,
    "fashionmnist": 1,
}


def get_model(modelname, dataset="", pretrained=None, latent_dim=10, **kwargs):
    model_fn = model_dict[modelname]
    num_classes = get_nclasses(dataset)

    if modelname in [
        "conv3",
        "lenet",
        "res20",
        "conv3_mnist",
    ]:
        model = model_fn(num_classes)

    elif modelname == "wres22":
        if dataset in ["mnist", "fashionmnist"]:
            model = model_fn(
                depth=22,
                num_classes=num_classes,
                widen_factor=2,
                dropRate=0.0,
                upsample=True,
                in_channels=1,
            )
        else:
            model = model_fn(
                depth=22, num_classes=num_classes, widen_factor=2, dropRate=0.0
            )
    elif modelname in ["conv3_gen"]:
        model = model_fn(
            z_dim=latent_dim,
            start_dim=gen_dim_dict[dataset],
            out_channels=gen_channels_dict[dataset],
        )
    elif modelname in ["conv3_cgen"]:
        model = model_fn(
            z_dim=latent_dim,
            start_dim=gen_dim_dict[dataset],
            out_channels=gen_channels_dict[dataset],
            n_classes=num_classes,
        )
    elif modelname in ["conv3_dis"]:
        model = model_fn(channels=gen_channels_dict[dataset], dataset=dataset)
    elif modelname in ["res18_ptm", "vgg13_bn"]:
        model = model_fn(pretrained=pretrained)
    else:
        sys.exit("unknown model")

    return model
