# -*- coding: utf-8 -*-

import gnumpy as gp
import numpy as np
import math
import decimal
from sys import stderr
from time import time

import rbm
import ml.common

from ml.common.util import logsum, logplus, logminus

class AnnealedImportanceSampler(object):
    
    def __init__(self, rbm, base_bias_vis=None):
        self.rbm = rbm
        self.base_bias_vis = base_bias_vis

    def init_using_sampling_from_rbm(self, n_samples, n_gibbs_chains, 
                                     gibbs_steps_between_samples):
        "Calculates the biases of the base rate RBM using samples from the RBM"
        n_steps = int(math.ceil(n_samples / float(n_gibbs_chains)))
        vis = self.rbm.sample_vis(n_chains, n_steps, 
                                  gibbs_steps_between_samples)

        self.init_using_dataset(vis)

    def init_using_dataset(self, vis_samples):
        "Calculates the biases of the base rate RBM using the given samples"
        epsilon = 1e-2
        vis_mean = gp.mean(vis_samples, axis=0)
        self.base_bias_vis = gp.log((vis_mean + epsilon) / (1 - vis_mean + epsilon))

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
        return part_vis * part_hid

    def base_log_partition_function(self):
        "Computes the log of the partition function of the base rate RBM"
        part_vis = gp.sum(gp.log_1_plus_exp(self.base_bias_vis))
        part_hid = self.rbm.n_hid * math.log(2)
        return part_vis + part_hid

    def log_partition_function(self, betas, ais_runs, sampling_gibbs_steps=1,
                               mean_precision=0):     
        "Computes the partition function of the RBM"      
        start_time = time()
        
        assert betas[0] == 0 and betas[-1] == 1

        irbm = ml.rbm.RestrictedBoltzmannMachine(0,
                                              self.rbm.n_vis,
                                              self.rbm.n_hid,
                                              0)
        iw = gp.zeros(ais_runs)

        for i, beta in enumerate(betas):
            #print >>stderr, "%d / %d                       \r" % (i, len(betas)),
            if ml.common.show_progress and i % 1000 == 0:
                print "%d / %d" % (i, len(betas))

            beta = float(beta)

            # calculate log p_(i-1)(v)
            if beta != 0:
                lp_prev_vis = -irbm.free_energy(vis)

            # build intermediate RBM
            irbm.weights = beta * self.rbm.weights
            irbm.bias_hid = beta * self.rbm.bias_hid
            irbm.bias_vis = ((1.0-beta) * self.base_bias_vis + 
                             beta * self.rbm.bias_vis)

            # calculate log p_i(v_i)
            if beta != 0:
                lp_vis = -irbm.free_energy(vis)

            # update importance weight
            if beta != 0:
                iw += lp_vis - lp_prev_vis

            # sample v_(i+1)
            if beta == 0:
                vis = self.base_sample_vis(ais_runs)
            else:
                vis, _ = irbm.gibbs_sample(vis, sampling_gibbs_steps)

        # calculate mean and standard deviation       
        if mean_precision == 0:
            npiw = gp.as_numpy_array(iw) 
            w = npiw + self.base_log_partition_function()
            wmean = logsum(w) - np.log(w.shape[0])
            wsqmean = logsum(2 * w) - np.log(w.shape[0])
            wstd = logminus(wsqmean, wmean) / 2.0
            wmeanstd = wstd - 0.5 * math.log(w.shape[0])
            
            wmean_plus_3_std = logplus(wmean, wmeanstd + math.log(3.0))
            wmean_minus_3_std = logminus(wmean, wmeanstd + math.log(3.0), 
                                         raise_when_negative=False)
        else:
            with decimal.localcontext() as ctx:
                ctx.prec = mean_precision
                blpf = self.base_log_partition_function()

                ewsum = decimal.Decimal(0)
                ewsqsum = decimal.Decimal(0)
                for w in gp.as_numpy_array(iw):
                    ew = decimal.Decimal(w + blpf).exp()
                    ewsum += ew
                    ewsqsum += ew ** 2
                ewmean = ewsum / iw.shape[0]
                ewsqmean = ewsqsum / iw.shape[0]
                ewstd = (ewsqmean - ewmean**2).sqrt()
                ewstdmean = ewstd / math.sqrt(iw.shape[0])

                wmean = float(ewmean.ln())
                wmean_plus_3_std = float((ewmean + 3*ewstdmean).ln())
                if ewmean - 3*ewstd > 0:
                    wmean_minus_3_std = float((ewmean - 3*ewstdmean).ln())
                else:
                    wmean_minus_3_std = float('-inf')

        end_time = time()
        print "AIS took %d s" % (end_time - start_time)

        return wmean, wmean_minus_3_std, wmean_plus_3_std


