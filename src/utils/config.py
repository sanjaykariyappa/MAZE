import argparse

parser = argparse.ArgumentParser(
    description="MAZE: Model Stealing attack using Zeroth order gradient Estimation"
)

# Shared
parser.add_argument(
    "--dataset",
    type=str,
    default="cifar10",
    help="mnist/fashionmnist/cifar10/cifar100/svhn/gtsrb",
)

parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=128, help="Batch Size")
parser.add_argument(
    "--logdir", type=str, default="./logs", help="Path to output directory"
)
parser.add_argument("--device", type=str, default="gpu", help="cpu/gpu/tpu")
parser.add_argument("--ngpu", type=int, default=1, help="Num GPUs to use")

# Defender
parser.add_argument(
    "--lr_tgt", type=float, default=0.1, help="Learning Rate of Target Model"
)
parser.add_argument(
    "--model_tgt", type=str, default="res20", help="Target Model[res20/wres22/conv3]"
)
parser.add_argument("--pretrained", action="store_true", help="Use Pretrained model")
parser.add_argument("--adv_train", action="store_true", help="Use adversarial training")
parser.add_argument("--eps_adv", type=float, default=0.1, help="epsilon for adv train")

# Attacker
parser.add_argument("--white_box", action="store_true", help="assume white box target")
parser.add_argument(
    "--attack",
    type=str,
    default="maze",
    help="Attack Type [knockoff/abm/zoo/knockoff_zoo/knockoff_augment]",
)
parser.add_argument("--opt", type=str, default="sgd", help="Optimizer [adam/sgd]")
parser.add_argument(
    "--model_clone", type=str, default="wres22", help="Clone Model [res20/wres22/conv3]"
)
parser.add_argument(
    "--model_gen", type=str, default="conv3_gen", help="Generator Model [conv3_gen]"
)
parser.add_argument(
    "--model_dis", type=str, default="conv3_dis", help="Discriminator Model [conv3_dis]"
)
parser.add_argument(
    "--lr_clone", type=float, default=0.1, help="Learning Rate of Clone Model"
)
parser.add_argument(
    "--latent_dim", type=int, default=100, help="dimensionality of latent vector"
)

# KnockoffNets
parser.add_argument(
    "--dataset_sur",
    type=str,
    default="cifar100",
    help="mnist/fashionmnist/cifar10/cifar100/svhn/gtsrb",
)

# MAZE
parser.add_argument(
    "--budget", type=float, default=5e6, metavar="N", help="Query Budget for Attack"
)
parser.add_argument(
    "--log_iter", type=float, default=5e5, metavar="N", help="log frequency"
)

parser.add_argument(
    "--lr_gen", type=float, default=1e-4, help="Learning Rate of Generator Model"
)
parser.add_argument(
    "--lr_dis", type=float, default=1e-4, help="Learning Rate of Discriminator Model"
)
parser.add_argument(
    "--eps", type=float, default=1e-3, help="Perturbation size for noise"
)
parser.add_argument(
    "--ndirs", type=int, default=10, help="Number of directions for MAZE"
)
parser.add_argument(
    "--mu", type=float, default=0.001, help="Smoothing parameter for MAZE"
)

parser.add_argument(
    "--iter_gen", type=int, default=1, help="Number of iterations of Generator"
)
parser.add_argument(
    "--iter_clone", type=int, default=5, help="Number of iterations of Clone"
)
parser.add_argument(
    "--iter_exp", type=int, default=10, help="Number of Exp Replay iterations of Clone"
)

parser.add_argument(
    "--lambda1", type=float, default=10, help="Gradient penalty multiplier"
)
parser.add_argument("--disable_pbar", action="store_true", help="disable progress bar")
parser.add_argument(
    "--alpha_gan", type=float, default=0.0, help="Weight given to gan term"
)


parser.add_argument(
    "--wandb_project", type=str, default="maze", help="wandb project name"
)


# JBDA
parser.add_argument(
    "--aug_rounds", type=int, default=6, help="Number of augmentation rounds for JBDA"
)
parser.add_argument(
    "--num_seed", type=int, default=100, help="Number of seed examples for JBDA"
)

# noise
parser.add_argument(
    "--noise_type",
    type=str,
    default="ising",
    choices=["ising", "uniform"],
    help="noise type",
)

# Extra
parser.add_argument(
    "--iter_gan", type=float, default=0, help="Number of iterations of Generator"
)
parser.add_argument(
    "--iter_log_gan",
    type=float,
    default=1e4,
    metavar="N",
    help="log frequency for GAN training",
)
parser.add_argument(
    "--iter_dis", type=int, default=5, help="Number of iterations of Discriminator"
)
parser.add_argument("--load_gan", action="store_true", help="load gan")
