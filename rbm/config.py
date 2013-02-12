# -*- coding: utf-8 -*-

import gnumpy as gp
import numpy as np

import util

class TrainingConfiguration(object):
    """Training configuration for RBM"""
    def __init__(self, dataset, n_vis, n_hid, batch_size, n_gibbs_steps, epochs,
                 step_rate,
                 use_pcd, binarize_data, initial_momentum, final_momentum,
                 use_final_momentum_from_epoch, weight_cost, 
                 init_method, init_weight_sigma, init_bias_sigma,
                 seed):
        self.dataset = dataset
        self.n_vis = n_vis
        self.n_hid = n_hid
        self.batch_size = batch_size
        self.n_gibbs_steps = n_gibbs_steps
        self.epochs = epochs
        self.step_rate = step_rate
        self.use_pcd = use_pcd
        self.binarize_data = binarize_data
        self.initial_momentum = initial_momentum
        self.final_momentum = final_momentum
        self.use_final_momentum_from_epoch = use_final_momentum_from_epoch
        self.weight_cost = weight_cost
        self.init_method = init_method
        self.init_weight_sigma = init_weight_sigma
        self.init_bias_sigma = init_bias_sigma
        self.seed = seed

        self.load_dataset()

    def load_dataset(self):
        if self.dataset == 'mnist':
            self.X, self.TX = util.load_mnist(False)
        elif self.dataset == 'mnistv':
            self.X, self.VX, self.TX = util.load_mnist(True)
        elif self.dataset == 'rmnist':
            self.X, self.TX = util.load_ruslan_mnist()

    @property
    def output_dir(self):
        if self.use_pcd:
            pcd_str = "p"
        else:
            pcd_str = ""
        if self.binarize_data:
            bin_str = "bin-"
        else:
            bin_str = ""
        return "%s-rbm-%03d-%scd%02d-mbs%04d-%ssr%.03f-m%.02f;%0.02f(%02d)-c%.04f-iws%.04f-ibs%.04f-%010d" % \
            (self.dataset, self.n_hid, pcd_str, self.n_gibbs_steps, self.batch_size,
             bin_str, self.step_rate, self.initial_momentum, self.final_momentum, 
             self.use_final_momentum_from_epoch, self.weight_cost, 
             self.init_weight_sigma, self.init_bias_sigma,
             self.seed)
