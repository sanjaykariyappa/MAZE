import torch.nn as nn
import wandb
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm

tanh = nn.Tanh()
from torch.autograd import Variable
from torch import autograd
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from collections import Counter
from datasets import get_nclasses


def gradient_penalty(fake_data, real_data, discriminator):
    alpha = (
        torch.cuda.FloatTensor(fake_data.shape[0], 1, 1, 1)
        .uniform_(0, 1)
        .expand(fake_data.shape)
    )
    interpolates = alpha * fake_data + (1 - alpha) * real_data
    interpolates.requires_grad = True
    disc_interpolates = discriminator(interpolates)

    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def consistency_term(real_data, discriminator, Mtag=0):
    d1, d_1 = discriminator(real_data)
    d2, d_2 = discriminator(real_data)

    # why max is needed when norm is positive?
    consistency_term = (
        (d1 - d2).norm(2, dim=1) + 0.1 * (d_1 - d_2).norm(2, dim=1) - Mtag
    )
    return consistency_term.mean()


def distill_epoch(T, S, device, train_loader, opt, disable_pbar=False):
    T.eval()
    S.train()
    correct = 0
    train_loss = 0
    criterion = torch.nn.KLDivLoss(reduction="batchmean")
    for batch_idx, (data, _) in enumerate(
        tqdm(train_loader, ncols=80, disable=disable_pbar, leave=False)
    ):
        data = data.to(device)
        with torch.no_grad():
            target = T(data)
            target = F.softmax(target, dim=1)
        opt.zero_grad()
        logits = S(data)
        preds_log = F.log_softmax(logits, dim=1)
        loss = criterion(preds_log, target)
        loss.backward()
        train_loss += loss
        opt.step()
        pred = logits.argmax(dim=1, keepdim=True)
        target = target.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader)
    train_acc = correct * 100.0 / len(train_loader.dataset)
    return train_loss, train_acc


def train_soft_epoch(model, device, train_loader, opt, disable_pbar=False):
    model.train()
    correct = 0
    train_loss = 0
    criterion = torch.nn.KLDivLoss(reduction="batchmean")
    for batch_idx, (data, target) in enumerate(
        tqdm(train_loader, ncols=80, disable=disable_pbar, leave=False)
    ):
        data, target = data.to(device), target.to(device)
        opt.zero_grad()
        logits = model(data)
        preds_log = F.log_softmax(logits, dim=1)
        loss = criterion(preds_log, target)
        loss.backward()
        train_loss += loss
        opt.step()
        pred = logits.argmax(dim=1, keepdim=True)
        target = target.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    train_loss /= len(train_loader)
    train_acc = correct * 100.0 / len(train_loader.dataset)
    return train_loss, train_acc


def gen_loss_noreduce(args, teacher_logits, student_logits):
    divergence = -F.kl_div(
        F.log_softmax(student_logits, dim=1),
        F.softmax(teacher_logits, dim=1),
        reduction="none",
    )  # forward KL
    return divergence.sum(dim=1)


def gen_target_loss_noreduce(args, teacher_logits, labels):
    divergence = -F.kl_div(
        F.log_softmax(student_logits, dim=1),
        F.softmax(teacher_logits, dim=1),
        reduction="none",
    )  # forward KL
    return divergence.sum(dim=1)


def kl_div_logits(args, teacher_logits, student_logits, reduction="batchmean"):
    divergence = F.kl_div(
        F.log_softmax(student_logits, dim=1),
        F.softmax(teacher_logits, dim=1),
        reduction=reduction,
    )  # forward KL
    return divergence


def sur_stats(logits_S, logits_T):
    pred_S = F.softmax(logits_S, dim=-1)
    pred_T = F.softmax(logits_T, dim=-1)
    mse = torch.nn.MSELoss()
    mse_loss = mse(pred_S, pred_T)
    max_diff = torch.max(torch.abs(pred_S - pred_T), dim=-1)[0]
    max_pred = torch.max(pred_T, dim=-1)[0]
    return mse_loss, max_diff.mean(), max_pred.mean()


def generate_images(args, G, z, labels=None, title="Generator Images"):

    n_classes = get_nclasses(args.dataset)
    if "cgen" in args.model_gen:
        x, _ = G(z, labels)
    else:
        x, _ = G(z)

    x_np = x.detach().cpu().numpy()
    x_np = np.moveaxis(x_np, 1, 2)
    x_np = np.moveaxis(x_np, 2, 3)
    fig, ax = plt.subplots(5, 5)

    for i in range(5):
        for j in range(5):
            if args.dataset in ["mnist", "fashionmnist", "brain"]:
                ax[i][j].imshow((x_np[(i * 5) + j, :, :, 0] + 1) / 2, cmap="gray")
            elif args.dataset in ["cifar10", "cifar100", "svhn", "diabetic5", "gtsrb"]:
                ax[i][j].imshow((x_np[(i * 5) + j, :, :, :] + 1) / 2)
            else:
                sys.exit("unknown dataset {}".format(args.dataset))
            ax[i][j].axis("off")

    wandb.log({title: fig})
    plt.close("all")
    return x


def generate_class_hist(pred_labels, title="Generator Labels"):
    wandb.log({title: wandb.Histogram(pred_labels)})


def zoge_target_backward(args, x_pre, labels, T):
    grad_est = torch.zeros_like(x_pre)
    d = np.array(x_pre.shape[1:]).prod()
    criterion = nn.CrossEntropyLoss()
    criterion_noreduce = nn.CrossEntropyLoss()

    with torch.no_grad():
        Tout = T(tanh(x_pre))
        lossG_target = criterion(Tout, labels)

        for _ in range(args.ndirs):
            u = torch.randn(x_pre.shape, device=args.device)
            u_flat = u.view([args.batch_size, -1])
            u_norm = u / torch.norm(u_flat, dim=1).view([-1, 1, 1, 1])
            x_mod_pre = x_pre + (args.mu * u_norm)
            Tout = T(tanh(x_mod_pre))
            lossG_target_mod = criterion_noreduce(Tout, labels)
            grad_est += (
                (d / args.ndirs) * (lossG_target_mod - lossG_target) / args.mu
            ).view([-1, 1, 1, 1]) * u_norm

    grad_est /= args.batch_size
    grad_est_flat = grad_est.view([args.batch_size, -1])

    x_det_pre = x_pre.detach()
    x_det_pre.requires_grad = True
    x_det_pre.retain_grad()
    Tout = T(tanh(x_det_pre))
    lossG_det = criterion(Tout, labels)
    lossG_det.backward()
    grad_true_flat = x_det_pre.grad.view([args.batch_size, -1])

    cos = nn.CosineSimilarity(dim=1)
    cs = cos(grad_true_flat, grad_est_flat)
    mag_ratio = grad_est_flat.norm(2, dim=1) / grad_true_flat.norm(2, dim=1)
    lossG = lossG_det.detach()
    return lossG.mean(), cs.mean(), mag_ratio.mean(), grad_est


def zoge_backward(args, x_pre, x, S, T):
    for p in S.parameters():
        p.requires_grad = False

    grad_est = torch.zeros_like(x_pre)
    d = np.array(x.shape[1:]).prod()

    with torch.no_grad():
        Sout = S(x)
        Tout = T(x)
        lossG = gen_loss_noreduce(args, Tout, Sout)
        for _ in range(args.ndirs):
            u = torch.randn(x_pre.shape, device=args.device)
            u_flat = u.view([args.batch_size, -1])
            u_norm = u / torch.norm(u_flat, dim=1).view([-1, 1, 1, 1])
            x_mod_pre = x_pre + (args.mu * u_norm)
            x_mod = tanh(x_mod_pre)
            Sout = S(x_mod)
            Tout = T(x_mod)
            lossG_mod = gen_loss_noreduce(args, Tout, Sout)
            grad_est += ((d / args.ndirs) * (lossG_mod - lossG) / args.mu).view(
                [-1, 1, 1, 1]
            ) * u_norm

    grad_est /= args.batch_size
    x_det_pre = x_pre.detach()
    x_det_pre.requires_grad = True
    x_det_pre.retain_grad()
    x_det = tanh(x_det_pre)
    Sout = S(x_det)
    Tout = T(x_det)
    lossG_det = -kl_div_logits(args, Tout, Sout)
    lossG_det.backward()
    grad_true_flat = x_det_pre.grad.view([args.batch_size, -1])
    grad_est_flat = grad_est.view([args.batch_size, -1])
    cos = nn.CosineSimilarity(dim=1)
    cs = cos(grad_true_flat, grad_est_flat)
    mag_ratio = grad_est_flat.norm(2, dim=1) / grad_true_flat.norm(2, dim=1)

    x_pre.backward(grad_est, retain_graph=True)

    for p in S.parameters():
        p.requires_grad = True

    lossG = lossG_det.detach()
    return lossG.mean(), cs.mean(), mag_ratio.mean()

