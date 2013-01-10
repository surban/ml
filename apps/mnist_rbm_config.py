# -*- coding: utf-8 -*-

# RBM model parameters
n_vis = 784
#n_hid = 512
n_hid = 25

# RBM training parameters
batch_size = 20
step_rate = 1E-1
#momentum = 0.9
momentum = 0
n_gibbs_steps = 3
#n_gibbs_steps = 15
epochs = 15
#use_pcd = True
use_pcd = False

# plotting parameters
n_plot_samples = 10
n_gibbs_steps_between_samples = 1000
