The deepClassifier_dynamics folder contains the original code for binary classifciation on the two classes of CIFAR-10 dtaset trained with "ground truth labels" and "random labels". 

* Experimentatl settings: 

training and testing data size are 10000, 2000, respectively; learning rate = 0.01, SGD optimizer, batch size =128, MSE loss, no data augmentation, initilizations = {0.01, 0.1, 0.5, 1, 3, 5, 10}, weight decay (WD)  = 0.01 or 0, using Batch Normalization (BN)

* Code file descriptions:

- train_binary.py -- the main code for binary classification experiments trained with the Square loss, BN and Weight Decay 0.01 or 0;
- extarget.py -- build and initialize our deep neural network model
- complexity.py -- compute the network complexity parameters (product norm -rho, batch norm std, etc.)
- final\_visualization\_0602.ipynb -- The visualization notebook file for plotting all figures from the saved tensorboard event data (see in the train_binary.py) in the paper.

* Main steps to run the code:

1) To run the random label experiment with different ratio by changing the parameter "ratio" as 0.2, 0.4, 0.6, 0.8 or 1.0 using the python script in the below:

python train_binary_RL.py --expDir ~/train_wd_RL20p/fig23_BN_noBN_weight_decay/binary_class_1_2_NetSimpleConv4_normx1_lr_d01_scale_d1_hasbn_1_decay_0.01/       --dataset cifar10 --class1 1 --class2 2 --layers 10 --widen-factor 4 --epochs 1000 --ratio 0.2 --init-scale 0.1 --exp-name 10K_n_wd --init-type const_norm --lr 0.01 --arch NetSimpleConv4 --weight-decay 0.01 --loss_type MSE --nesterov 0 --no-augment --tensorboard --normx1 L2  --hasbn 1

2) To run the binary classification experiments with MSE loss, BN and WD or w/o WD

# train the network with BN (hasbn 1) + WD (weight-decay 0.01) + initialization (init-scale) 0.01

python train_binary.py --expDir ~/train_wd_random_labels/fig23_BN_noBN_weight_decay/binary_class_1_2_NetSimpleConv4_normx1_lr_d01_scale_d01_hasbn_1_decay_0.01/       --dataset cifar10 --class1 1 --class2 2 --layers 10 --widen-factor 4 --epochs 1000 --init-scale 0.01 --exp-name 10K_n_wd --init-type const_norm --lr 0.01 --arch NetSimpleConv4 --weight-decay 0.01 --loss_type MSE --nesterov 0 --no-augment --tensorboard --normx1 L2  --hasbn 1


# train the network with BN + w/o WD (weight-decay 0) + initialization (init-scale) 0.01

python train_binary.py --expDir ~/train_wd_random_labels/fig23_BN_noBN_weight_decay/binary_class_1_2_NetSimpleConv4_normx1_lr_d01_scale_d01_hasbn_1_decay_0.01/       --dataset cifar10 --class1 1 --class2 2 --layers 10 --widen-factor 4 --epochs 1000 --init-scale 0.01 --exp-name 10K_n_wd --init-type const_norm --lr 0.01 --arch NetSimpleConv4 --weight-decay 0.01 --loss_type MSE --nesterov 0 --no-augment --tensorboard --normx1 L2  --hasbn 1

3) Try to use the scripts in the jupyter notebook file (*.ipynb) to visualize the related results in the paper.
