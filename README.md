### This repo contains the original implementations for our paper [``Normalization and dynamics in Deep Classifiers trained with the Square Loss''](https://cbmm.mit.edu/sites/default/files/publications/JMLR__2021-22.pdf) in PyTorch
<a href="#"><img src="https://img.shields.io/github/workflow/status/milesial/PyTorch-UNet/Publish%20Docker%20image?logo=github&style=for-the-badge" /></a>
<a href="https://hub.docker.com/r/milesial/unet"><img src="https://img.shields.io/badge/docker%20image-available-blue?logo=Docker&style=for-the-badge" /></a>
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-v1.9.0-red.svg?logo=PyTorch&style=for-the-badge" /></a>
<a href="#"><img src="https://img.shields.io/badge/python-v3.6+-blue.svg?logo=python&style=for-the-badge" /></a>

![training dynamics of deep nets](https://github.com/GraceXu182/binaryclassifiers_squareloss/blob/main/ynfn_rho_over_1000iterations_github.png)

- [Main file descriptions](#main-file-descriptions)
- [Experimental settings](#experimental-settings)
- [How to run the code?](#how-to-run-the-code)

## Main file descriptions

- [train_binary.py](https://github.com/GraceXu182/binaryclassifiers_squareloss/blob/0b088bab2295f8ac75dd8bda202bbbe2571aa72e/train_binary.py) -- the main code for binary classification experiments trained with the Square loss, BN and Weight Decay 0.01 or 0;
- [extarget.py](https://github.com/GraceXu182/binaryclassifiers_squareloss/blob/0b088bab2295f8ac75dd8bda202bbbe2571aa72e/extarget.py) -- build and initialize our deep neural network model
- [complexity.py](https://github.com/GraceXu182/binaryclassifiers_squareloss/blob/0b088bab2295f8ac75dd8bda202bbbe2571aa72e/complexity.py) -- compute the network complexity parameters (product norm -rho, batch norm std, etc.)
- [final\_visualization\_0602.ipynb](https://www.dropbox.com/s/717k1cug1ejxqcn/final_visualization_0602.zip?dl=0) -- The visualization notebook file for plotting all figures added in our paper from the saved tensorboard event files (see in the train_binary.py).

## Experimental settings

In the experiments, we train the binary classifier with mean square loss, 10000 images are used for training and 2000 ones for testing. The detailed hyperparameters include learning rate 0.01, SGD optimizer, batch size 128, mean square loss, no data augmentation, initilization scales options {0.01, 0.1, 0.5, 1, 3, 5, 10}, weight decay (WD) 0.01 or 0, and Batch Normalization (BN).

## How to run the code?

* To run the random label experiment with different ratio by changing the parameter "ratio" as 0.2, 0.4, 0.6, 0.8 or 1.0 using the python script in the below:
```bash
python train_binary_RL.py --expDir ~/train_wd_RL20p/fig23_BN_noBN_weight_decay/binary_class_1_2_NetSimpleConv4_normx1_lr_d01_scale_d1_hasbn_1_decay_0.01/       --dataset cifar10 --class1 1 --class2 2 --layers 10 --widen-factor 4 --epochs 1000 --ratio 0.2 --init-scale 0.1 --exp-name 10K_n_wd --init-type const_norm --lr 0.01 --arch NetSimpleConv4 --weight-decay 0.01 --loss_type MSE --nesterov 0 --no-augment --tensorboard --normx1 L2  --hasbn 1
```
* To run the binary classification experiments with MSE loss, BN and WD or w/o WD

 1) train the network with BN (hasbn 1) + WD (weight-decay 0.01) + initialization (init-scale) 0.01
```bash
python train_binary.py --expDir ~/train_wd_random_labels/fig23_BN_noBN_weight_decay/binary_class_1_2_NetSimpleConv4_normx1_lr_d01_scale_d01_hasbn_1_decay_0.01/       --dataset cifar10 --class1 1 --class2 2 --layers 10 --widen-factor 4 --epochs 1000 --init-scale 0.01 --exp-name 10K_n_wd --init-type const_norm --lr 0.01 --arch NetSimpleConv4 --weight-decay 0.01 --loss_type MSE --nesterov 0 --no-augment --tensorboard --normx1 L2  --hasbn 1
```

 2) train the network with BN + w/o WD (weight-decay 0) + initialization (init-scale) 0.01
```bash
python train_binary.py --expDir ~/train_wd_random_labels/fig23_BN_noBN_weight_decay/binary_class_1_2_NetSimpleConv4_normx1_lr_d01_scale_d01_hasbn_1_decay_0.01/       --dataset cifar10 --class1 1 --class2 2 --layers 10 --widen-factor 4 --epochs 1000 --init-scale 0.01 --exp-name 10K_n_wd --init-type const_norm --lr 0.01 --arch NetSimpleConv4 --weight-decay 0.01 --loss_type MSE --nesterov 0 --no-augment --tensorboard --normx1 L2  --hasbn 1
```
 3) If you want to visualize the experimental results, please use the scripts in the jupyter notebook file (*.ipynb)
