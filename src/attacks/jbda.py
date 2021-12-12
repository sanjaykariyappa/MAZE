import torch
from torch.autograd import Variable
import numpy as np 
import torch.optim as optim
from utils.helpers import test
import torch.nn.functional as F

num_classes_dict={
    'fashionmnist':10,
    'cifar10':10,
    'cifar100':100,
    'svhn':10,
    'gtsrb':43
}

def get_labels(X_sub, blackbox):
    scores = []
    label_batch = 64
    X_sub_len = X_sub.shape[0]
    num_splits = 1 + int(X_sub_len / label_batch)
    splits = np.array_split(X_sub, num_splits, axis=0)

    for x_sub in splits:
        score_batch = blackbox(to_var(torch.from_numpy(x_sub)))
        score_batch = F.softmax(score_batch, dim=1)
        score_batch = score_batch.data.cpu().numpy()
        scores.append(score_batch)
    scores = np.concatenate(scores)
    print('done labeling')

    y_sub = scores
    return y_sub

def batch_indices(batch_nb, data_length, batch_size):
    # Batch start and end index
    start = int(batch_nb * batch_size)
    end = int((batch_nb + 1) * batch_size)

    # When there are not enough inputs left, we reuse some to complete the
    # batch
    if end > data_length:
        shift = end - data_length
        start -= shift
        end -= shift

    return start, end

def to_var(x, requires_grad=False, volatile=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)

def jacobian(model, x, nb_classes=10):
    """
    This function will return a list of PyTorch gradients
    """
    list_derivatives = []
    x_var = to_var(torch.from_numpy(x), requires_grad=True)

    # derivatives for each class
    for class_ind in range(nb_classes):
        x_var_exp = x_var.unsqueeze(0)
        score = model(x_var_exp)[:, class_ind]
        score.backward()
        list_derivatives.append(x_var.grad.data.cpu().numpy())
        x_var.grad.data.zero_()

    return list_derivatives


def jacobian_augmentation(model, X_sub_prev, Y_sub, lmbda=0.1, nb_classes=10):
    """
    Create new numpy array for adversary training data
    with twice as many components on the first dimension.
    """
    X_sub = np.vstack([X_sub_prev, X_sub_prev])
    if Y_sub.ndim == 2:
        # Labels could be a posterior probability distribution. Use argmax as a proxy.
        Y_sub = np.argmax(Y_sub, axis=1)

    # For each input in the previous' substitute training iteration
    offset = len(X_sub_prev)
    for ind, x in enumerate(X_sub_prev):
        grads = jacobian(model, x, nb_classes)
        # Select gradient corresponding to the label predicted by the oracle
        grad = grads[Y_sub[ind]]

        # Compute sign matrix
       	grad_val = np.sign(grad)

        # Create new synthetic point in adversary substitute training set
        X_sub[offset+ind] = x + lmbda * grad_val

    X_sub = np.clip(X_sub, -1, 1)


    # Return augmented training data (needs to be labeled afterwards)
    return X_sub


def jbda(args, T, S, train_loader, test_loader, tar_acc):

    train_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=args.num_seed, shuffle=True)
    #  Label seed data
    num_classes = num_classes_dict[args.dataset]
    data_iter = iter(train_loader)
    X_sub, _ = data_iter.next()
    X_sub = X_sub.numpy()
    y_sub = get_labels(X_sub, T)
    rng = np.random.RandomState()
    criterion = torch.nn.KLDivLoss(reduction='batchmean')

    optS = optim.Adam(S.parameters(), lr=args.lr_clone)

    # Train the substitute and augment dataset alternatively
    for aug_round in range(args.aug_rounds):
        # model training
        # Indices to shuffle training set
        index_shuf = list(range(len(X_sub)))
        rng.shuffle(index_shuf)

        for epoch in range(args.epochs):
            nb_batches = int(np.ceil(float(len(X_sub)) / args.batch_size))
            assert nb_batches * args.batch_size >= len(X_sub)

            for batch in range(nb_batches):
                start, end = batch_indices(batch, len(X_sub), args.batch_size)
                x = X_sub[index_shuf[start:end]]
                y = y_sub[index_shuf[start:end]]
                Sout = S(to_var(torch.from_numpy(x)))
                Sout = F.softmax(Sout, dim=1)
                lossS = criterion(Sout, to_var(torch.from_numpy(y)))
                optS.zero_grad()
                lossS.backward()
                optS.step()
            test_loss, test_acc = test(S, args.device, test_loader)

        # If we are not in the last substitute training iteration, augment dataset
        if aug_round < args.aug_rounds - 1:
            print("[{}] Augmenting substitute training data.".format(aug_round))
            # Perform the Jacobian augmentation
            X_sub = jacobian_augmentation(S, X_sub, y_sub, nb_classes=num_classes)

            print("Labeling substitute training data.")
            # Label the newly generated synthetic points using the black-box
            y_sub= get_labels(X_sub, T)
        print('Aug Round {} Clone Accuracy: {:.2f}({:.2f})x'.format(aug_round, test_acc, test_acc/tar_acc))