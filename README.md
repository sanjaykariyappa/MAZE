# MAZE: Data-Free Model Stealing Attack Using Zeroth-Order Gradient Estimation

Implementation of the paper "MAZE: Data-Free Model Stealing Attack Using Zeroth-Order Gradient Estimation".


# Usage

Here is the example usage for the FashionMnist dataset

## Train the defender model

python src/defender.py --dataset=fashionmnist --model_tgt=lenet --epochs=20


## Launch the attack

python src/attacker.py --dataset=fashionmnist --model_tgt=lenet --model_clone=wres22 --attack=maze --budget=5e6 --log_iter=1e5 --lr_clone=0.1 --lr_gen=1e-3 --iter_clone=5 --iter_exp=10




