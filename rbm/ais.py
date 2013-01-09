# -*- coding: utf-8 -*-

import gnumpy as gp
import numpy as np

#import .rbm

class AnnealedImportanceSampler(object):
    
    def __init__(self, rbm, batch_size, intermediate_steps, 
                 sampling_gibbs_steps,
                 init_n_chains, init_n_steps, 
                 init_gibbs_steps_between_samples):
        self.rbm = rbm
        self.batch_size = batch_size
        self.intermediate_steps = intermediate_steps
        self.sampling_gibbs_steps = sampling_gibbs_steps

        self.base_init(init_n_chains, init_n_steps,
                       init_gibbs_steps_between_samples)    

    def base_p_vis(self, vis):
        "Probability of visible units in base rate RBM"
        punit = (gp.exp(gp.dot(self.base_bias_vis, vis)) / 
                 (1 + gp.exp(self.base_bias_vis)))
        return gp.prod(punit, axis=1)

    def base_sample_vis(self):
        "Samples the visible units from the base rate RBM"
        p = gp.logistic(self.base_bias_vis)
        r = gp.rand((self.batch_size, self.base_bias_vis.shape[0]))
        return r < p

    def base_partition_function(self):
        "Computes the partition function of the base rate RBM"
        return gp.power(2, gp.prod(1 + gp.exp(self.base_bias_vis)))

    def base_init(self, n_chains, n_steps, gibbs_steps_between_samples):
        "Calculates the biases of the base rate RBM using maximum likelihood"
        vis = self.rbm.sample_free_vis(n_chains, n_steps, 
                                       gibbs_steps_between_samples)
        vis_mean = gp.mean(vis, axis=0)
        self.base_bias_vis = gp.log(vis_mean / (1 - vis_mean))

    def partition_function(self):     
        "Computes the partition function of the RBM"
        irbm = rbm.RestrictedBoltzmannMachine(self.batch_size,
                                              self.rbm.n_vis,
                                              self.rbm.n_hid,
                                              0)
        iw = gp.ones(self.batch_size)

        for beta in np.linspace(0, 1, self.intermedia_steps):
            # build intermediate RBM
            irbm.weights = beta * self.rbm.weights
            irbm.bias_hid = beta * self.rbm.bias_hid
            irbm.bias_vis = ((1-beta) * self.base_bias_vis + beta * 
                                beta * self.rbm.bias_vis)

            # sample
            if beta == 0:
                vis = self.base_sample_vis()
            else:
                vis = irbm.gibbs_sample(vis, self.sampling_gibbs_steps)

            # calculate unnormalized probability of visible units
            p = gp.exp(irbm.free_energy(vis))

            # update importance weight
            if beta != 0:
                iw = iw * p / last_p

            last_p = p

        return gp.mean(iw) * self.base_partition_function()

