import os
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import wandb
import random
import numpy as np
seed = 2020
random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.multiprocessing.set_sharing_strategy('file_system')

from utils.simutils.timer import timer
from utils.config import parser
from models import get_model
from datasets import get_dataset
from utils.helpers import test, train_epoch

args = parser.parse_args()

wandb.init(project=args.wandb_project)
run_name = 'defender_{}_{}'.format(args.dataset, args.model_tgt)
wandb.run.name = run_name
wandb.run.save()

if args.device == 'gpu':
    import torch.backends.cudnn as cudnn
    cudnn.enabled = True
    cudnn.benchmark = True
    args.device = 'cuda'
else:
    args.device = 'cpu'


def train_defender():
    model = get_model(args.model_tgt, args.dataset, args.pretrained)
    model = model.to(args.device)
    train_loader, test_loader = get_dataset(args.dataset, args.batch_size, augment=True)

    savedir = '{}/{}/{}/'.format(args.logdir, args.dataset, args.model_tgt)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    savepath = savedir + 'T.pt'

    sch = None
    if args.opt == 'sgd': # Paper uses SGD with cosine annealing for CIFAR10
        opt = optim.SGD(model.parameters(), lr=args.lr_tgt, momentum=0.9, weight_decay=5e-4)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, last_epoch=-1)
    elif args.opt == 'adam': # and Adam for the rest
        opt = optim.Adam(model.parameters(), lr=args.lr_tgt)
    else:
        sys.exit('Invalid optimizer {}'.format(args.opt))

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, args.device, train_loader, opt, args)
        test_loss, test_acc = test(model, args.device, test_loader)
        print('Epoch: {} Loss: {:.4f} Train Acc: {:.2f}% Test Acc: {:.2f}%\n'.format(epoch+1, train_loss, train_acc, test_acc))
        wandb.log({'Train Acc': train_acc, 'Test Acc': test_acc, "Train Loss": train_loss})
        if sch:
            sch.step()

    torch.save(model.state_dict(), savepath)

def main():
    timer(train_defender)
    exit(0)

if __name__ == '__main__':
    main()

