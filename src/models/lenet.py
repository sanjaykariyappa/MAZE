import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib
import numpy as np

class LeNet(nn.Module):
    """A simple MNIST network

    Source: https://github.com/pytorch/examples/blob/master/mnist/main.py
    """
    def __init__(self, num_classes=10, **kwargs):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 50, 5, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(4 * 4 * 50, 500),
            nn.ReLU(),
            nn.Linear(500, num_classes)
        )

        self.test_adv = False
        self.delta = 0.9
        self.adv_mode = 1
        self.adv_tensor = np.zeros([1, 10])
        self.adv_tensor[0, 5] = 1
        self.adv_tensor = torch.tensor(self.adv_tensor)
        self.input_count = 0
        self.ood_count = 0

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 4 * 4 * 50)
        x = self.classifier(x)

        if self.test_adv:
            probs = F.softmax(x, dim=1)  # batch x 10
            probs_max, _ = torch.max(probs, dim=1)  # batch
            batch = probs_max.size()[0]
            self.input_count += batch
            mask = probs_max > self.delta
            mask = mask.unsqueeze(dim=1)
            mask_not = ~mask
            self.ood_count += np.sum(mask_not.cpu().detach().numpy())
            if self.adv_mode == 1:
                shift = np.random.randint(1, 10)
                x_rand = torch.cat((x[:, -shift:], x[:, :-shift]), dim=1)  # rotation #DBG
            elif self.adv_mode == 2:
                shift = 1
                x_rand = torch.cat((x[:, -shift:], x[:, :-shift]), dim=1)  # rotation #DBG
            elif self.adv_mode == 3:
                self.adv_tensor = self.adv_tensor.to(x.device)
                x_rand = self.adv_tensor.repeat([batch, 1])
            x_adv = (mask.float() * x) + (mask_not.float() * x_rand.float())
            return x_adv
        else:
            return x

def lenet(num_classes, **kwargs):
    return LeNet(num_classes, **kwargs)
