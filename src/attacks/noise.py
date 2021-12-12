from tqdm import tqdm
import os
from torch.autograd import Variable
from scipy.stats import truncnorm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import test
from .attack_utils import kl_div_logits, generate_images, sur_stats
import wandb
from models import get_model
import pandas as pd
from utils.simutils import logs
import itertools
tanh = nn.Tanh()
import matplotlib
matplotlib.use('Agg')
import numpy as np
from numpy.random import randint, randn, rand

def noise(args, T, S, test_loader, tar_acc):
    T.eval(), S.train()
    schS  = None
    budget_per_iter = args.batch_size * (args.iter_clone)
    iter = int(args.budget / budget_per_iter)

    if args.opt == 'sgd':
        optS = optim.SGD(S.parameters(), lr=args.lr_clone, momentum=0.9, weight_decay=5e-4)
        schS = optim.lr_scheduler.CosineAnnealingLR(optS, iter, last_epoch=-1)
    else:
        optS = optim.Adam(S.parameters(), lr=args.lr_clone)
        #schS = optim.lr_scheduler.CosineAnnealingLR(optS, iter, last_epoch=-1)

    print('\n== Training Clone Model ==')

    pbar = tqdm(range(1, iter + 1), ncols=80, disable=args.disable_pbar, leave=False)
    query_count = 0
    log = logs.BatchLogs()
    start = time.time()
    results = {'queries': [], 'accuracy': [], 'accuracy_x': []}
    ds = []

    if 'mnist' in args.dataset:
        x_shape = [args.batch_size,1,28,28]
    else:
        x_shape = [args.batch_size,3,32,32]


    for p in T.parameters():
        p.requires_grad = False

    for i in pbar:

        ############################
        # (1) Update Clone network
        ###########################

        for c in range(args.iter_clone):

            if args.noise_type == 'uniform':
                x = (torch.rand(x_shape, device=args.device)-0.5)*2
            else:
                sys.exit(f'Unknown noise type {args.noise_type}')


            with torch.no_grad():
                Tout = T(x)

            Sout = S(x)
            lossS = kl_div_logits(args, Tout, Sout)
            optS.zero_grad()
            lossS.backward()
            optS.step()

        _, max_diff, max_pred = sur_stats(Sout, Tout)
        log.append_tensor(['Sur_loss', 'Max_diff', 'Max_pred'], [lossS, max_diff, max_pred])

        query_count += budget_per_iter

        if (query_count % args.log_iter < budget_per_iter and query_count > budget_per_iter) or i == iter:

            log.flatten()
            _, log.metric_dict['Sur_acc'] = test(S, args.device, test_loader)
            tar_acc_fraction = log.metric_dict['Sur_acc'] / tar_acc
            log.metric_dict['Sur_acc(x)'] = tar_acc_fraction

            metric_dict = log.metric_dict
            pbar.clear()
            time_100iter = int(time.time() - start)

            iter_M = (query_count / 1e6)
            print('Queries: {:.2f}M Losses:Sur {:.2f} Acc: Sur {:.2f} ({:.2f}x) time: {: d}'.format(iter_M,
                    metric_dict['Sur_loss'], metric_dict['Sur_acc'], tar_acc_fraction, time_100iter))

            wandb.log(log.metric_dict)
            results['queries'].append(iter_M)
            results['accuracy'].append(metric_dict['Sur_acc'])
            results['accuracy_x'].append(tar_acc_fraction)

            log = logs.BatchLogs()
            S.train()
            start = time.time()

        if schS:
            schS.step()

    savedir = '{}/{}/{}/'.format(args.logdir, args.dataset, args.runid)
    savedir_csv = savedir + 'csv/'
    if not os.path.exists(savedir_csv):
        os.makedirs(savedir_csv)

    df = pd.DataFrame(data=results)
    df.to_csv(savedir + 'noise.csv')
    return log.metric_dict

