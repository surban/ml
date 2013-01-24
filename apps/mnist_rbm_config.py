# -*- coding: utf-8 -*-

# RBM model parameters
n_vis = 784
#n_hid = 512
#n_hid = 25
n_hid = 25
#n_hid = 16
#n_hid = 8

# RBM training parameters
batch_size = 100
step_rate = 0.05
initial_momentum = 0.5
final_momentum = 0.9
use_final_momentum_from_epoch = 5

#n_gibbs_steps = 1
n_gibbs_steps = 3
#n_gibbs_steps = 15
epochs = 10
#epochs = 3
#use_pcd = True
use_pcd = False

# plotting parameters
n_plot_samples = 10
n_gibbs_steps_between_samples = 1000
