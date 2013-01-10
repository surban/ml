# -*- coding: utf-8 -*-

import gnumpy as gp
import numpy as np
import math

import rbm

class AnnealedImportanceSampler(object):
    
    def __init__(self, rbm,
                 init_n_samples, 
                 init_n_gibbs_chains, 
                 init_gibbs_steps_between_samples):
        self.rbm = rbm
        self.base_init(init_n_gibbs_chains, 
                       int(math.ceil(init_n_samples / init_n_gibbs_chains)),
                       init_gibbs_steps_between_samples)    

    def base_p_vis(self, vis):
        "Probability of visible units in base rate RBM"
        punit = (gp.exp(gp.dot(self.base_bias_vis, vis)) / 
                 (1 + gp.exp(self.base_bias_vis)))
        return gp.prod(punit, axis=1)

    def base_sample_vis(self, n_samples):
        "Samples the visible units from the base rate RBM"
        p = gp.logistic(self.base_bias_vis)
        r = gp.rand((n_samples, self.base_bias_vis.shape[0]))
        return r < p

    def base_partition_function(self):
        "Computes the partition function of the base rate RBM"
        part_vis = gp.prod(1 + gp.exp(self.base_bias_vis))
        part_hid = 2**self.rbm.n_hid
        print part_vis
        print part_hid
        return part_vis * part_hid

    def base_log_partition_function(self):
        "Computes the log of the partition function of the base rate RBM"
        part_vis = gp.sum(gp.log_1_plus_exp(self.base_bias_vis))
        part_hid = self.rbm.n_hid * math.log(2)
        return part_vis + part_hid

    def base_init(self, n_chains, n_steps, gibbs_steps_between_samples):
        "Calculates the biases of the base rate RBM using maximum likelihood"
        epsilon = 1e-2
        vis = self.rbm.sample_vis(n_chains, n_steps, 
                                       gibbs_steps_between_samples)
        vis_mean = gp.mean(vis, axis=0)
        self.base_bias_vis = gp.log((vis_mean + epsilon) / (1 - vis_mean + epsilon))

    def log_partition_function(self, betas, ais_runs, sampling_gibbs_steps=1):     
        "Computes the partition function of the RBM"
        assert betas[0] == 0 and betas[-1] == 1

        irbm = rbm.RestrictedBoltzmannMachine(0,
                                              self.rbm.n_vis,
                                              self.rbm.n_hid,
                                              0)
        iw = gp.zeros(ais_runs)

        for i, beta in enumerate(betas):
            print "%d / %d                       \r" % (i, len(betas)),

            beta = float(beta)

            # calculate log p_(i-1)(v)
            if beta != 0:
                lp_prev_vis = irbm.free_energy(vis)

            # build intermediate RBM
            irbm.weights = beta * self.rbm.weights
            irbm.bias_hid = beta * self.rbm.bias_hid
            irbm.bias_vis = ((1.0-beta) * self.base_bias_vis + 
                             beta * self.rbm.bias_vis)

            # calculate log p_i(v_i)
            if beta != 0:
                lp_vis = irbm.free_energy(vis)

            # update importance weight
            if beta != 0:
                iw += lp_vis - lp_prev_vis

            # sample v_(i+1)
            if beta == 0:
                vis = self.base_sample_vis(ais_runs)
            else:
                vis, _ = irbm.gibbs_sample(vis, sampling_gibbs_steps)

        return gp.mean(iw) + self.base_log_partition_function()

