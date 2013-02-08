# -*- coding: utf-8 -*-

import gnumpy as gp
import numpy as np
import common.util
import decimal
from sys import stderr

from .util import sample_binomial, all_states, save_parameters, plot_weights, \
    plot_pcd_chains
from common.util import logsum, draw_slices


class RestrictedBoltzmannMachine(object):
    "A Restricted Boltzmann Machine (RBM) with binary units"

    def __init__(self, batch_size, n_vis, n_hid, cd_steps, 
                 init_weight_sigma, init_bias_sigma):
        """A Restricted Boltzmann Machine (RBM) with binary units.
        batch_size is the size of a batch used for training. It may be 0 if
        the RBM will not be trained.
        n_vis and n_hid are the number of visible and hidden units.
        cd_steps is the number of alternating Gibbs sampling iterations to
        perform when computing the paramter updates using constrastive divergence."""
        self.cd_steps = cd_steps
        self.weights = gp.as_garray(np.random.normal(0, init_weight_sigma, 
                                                     size=(n_vis, n_hid)))
        if init_bias_sigma > 0:
            self.bias_vis = gp.as_garray(np.random.normal(0, init_bias_sigma, 
                                                          size=(n_vis,)))
            self.bias_hid = gp.as_garray(np.random.normal(0, init_bias_sigma, 
                                                          size=(n_hid,)))
        else:
            self.bias_vis = gp.zeros((n_vis,))
            self.bias_hid = gp.zeros((n_hid,))
        if batch_size > 0:
            self.persistent_vis = \
                gp.as_garray(np.random.normal(0, 1, size=(batch_size, n_vis)))
        self.log_pf = None

    @property
    def n_vis(self):
        return self.bias_vis.shape[0]

    @property
    def n_hid(self):
        return self.bias_hid.shape[0]

    def normalized_log_p_vis(self, vis):
        """Calculates the log probability of the visible units being in state vis
        (with the normalization constant)"""
        return - self.free_energy(vis) - self.log_pf

    def free_energy(self, vis):
        """The negative log probability of the visible units being in state vis 
        (without the normalization constant)"""
        return (- gp.dot(vis, self.bias_vis) 
                - gp.sum(gp.log_1_plus_exp(self.bias_hid + gp.dot(vis, self.weights)),
                         axis=1))

    def free_hidden_energy(self, hid):
        """The negative log probability of the hidden units being in state hid
        (without the normalization constant)"""
        return (- gp.dot(hid, self.bias_hid)
                - gp.sum(gp.log_1_plus_exp(self.bias_vis + gp.dot(hid, self.weights.T)),
                         axis=1))

    def partition_function(self, batch_size, prec):
        """The exact value of Z calculated with precision prec. 
        Only feasible for small number of hidden units."""
        with decimal.localcontext() as ctx:
            if prec != 0:
                ctx.prec = prec
            batches = common.util.pack_in_batches(all_states(self.n_hid), 
                                                  batch_size)
            if prec != 0:
                s = decimal.Decimal(0)
            else:
                allfhes = np.array([])
            seen_samples = 0L
            total_samples = 2L**self.n_hid
            for hid in batches:
                print >>stderr, "%i / %i           \r" % (seen_samples, total_samples),
                fhes = self.free_hidden_energy(hid)
                if prec != 0:
                    for fhe in gp.as_numpy_array(fhes):
                        p = decimal.Decimal(-fhe).exp()
                        s += p
                else:
                    allfhes = np.concatenate((allfhes, 
                                              -gp.as_numpy_array(fhes)))
                seen_samples += hid.shape[0]
            if prec != 0:
                return s
            else:
                return logsum(allfhes)

    def p_hid_given_vis(self, vis):
        """Returns a vector whose ith component is the probability that the ith
        hidden unit is active given the states of the visible units"""
        return gp.logistic(gp.dot(vis, self.weights) + self.bias_hid)

        #try:
        #    return gp.logistic(gp.dot(vis, self.weights) + self.bias_hid)
        #except Exception, e:
        #    np.set_printoptions(threshold=np.nan)
        #    print e
        #    print "vis: ", vis
        #    #print "weights: ", self.weights
        #    print "bias_hid: ", self.bias_hid
        #    print "prod: ", gp.dot(vis, self.weights) + self.bias_hid
        #    raise


    def sample_hid_given_vis(self, vis):
        """Samples the hidden units given the visible units"""
        p = self.p_hid_given_vis(vis)
        return sample_binomial(p)

    def p_vis_given_hid(self, hid):
        """Returns a vector whose ith component is the probability that the ith
        visible unit is active given the states of the hidden units"""
        return gp.logistic(gp.dot(hid, self.weights.T) + self.bias_vis)

    def sample_vis_given_hid(self, hid):
        """Samples the visible units given the hidden units"""
        p = self.p_vis_given_hid(hid)
        return sample_binomial(p)

    def gibbs_sample(self, vis, k):
        """Performs k steps of alternating Gibbs sampling. Returns a tuple
        consisting of the final state of the visible units and the probability
        that they are active given the state of the hiddens in the previous
        to last step."""
        for i in range(k):
            hid = self.sample_hid_given_vis(vis)
            vis = self.sample_vis_given_hid(hid)
        p_vis = self.p_vis_given_hid(hid)
        return (vis, p_vis)

    def sample_vis(self, n_chains, n_steps, gibbs_steps_between_samples,
                   sample_probabilities=False):
        """Obtains unbiased samples for the visible units.
        Runs n_chains Gibbs chains in parallel for n_steps.
        Grabs samples every gibbs_steps_between_samples Gibbs steps."""
        samples = gp.zeros((n_chains * n_steps, self.n_vis))
        vis = gp.rand((n_chains, self.n_vis)) < 0.5
        for step in range(n_steps):
            print >>stderr, "%d / %d                 \r" % (step, n_steps),
            vis, p_vis = self.gibbs_sample(vis, gibbs_steps_between_samples)
            if sample_probabilities:
                sample = p_vis
            else:
                sample = vis
            samples[step*n_chains : (step+1)*n_chains, :] = sample
        return samples

    def _cd_update_terms(self, vis, model_vis, model_p_vis):
        """Returns (weights update, visible bias update, hidden bias update) given
        visible states from the data vis, visible states sampled from the 
        model model_vis and the probability of the visible units being active         
        from the model."""
        #print "vis.shape:                ", vis.shape
        #print "p_hid(vis).shape:         ", self.p_hid(vis).shape
        #print "model_p_vis.shape:        ", model_p_vis.shape
        #print "p_hid(model_p_vis).shape: ", self.p_hid(model_p_vis).shape
        
        # my update rule:
        #dweights = (gp.dot(vis.T, self.p_hid(vis)) - 
        #            gp.dot(model_p_vis.T, self.p_hid(model_vis)))
        #dbias_vis = gp.sum(vis, axis=0) - gp.sum(model_p_vis, axis=0)
        #dbias_hid = (gp.sum(self.p_hid(vis), axis=0) - 
        #             gp.sum(self.p_hid(model_vis), axis=0))

        # deep learning update rule:
        dweights = (gp.dot(vis.T, self.p_hid_given_vis(vis)) - 
                    gp.dot(model_vis.T, self.p_hid_given_vis(model_vis)))
        dbias_vis = gp.sum(vis, axis=0) - gp.sum(model_vis, axis=0)
        dbias_hid = (gp.sum(self.p_hid_given_vis(vis), axis=0) - 
                     gp.sum(self.p_hid_given_vis(model_vis), axis=0))

        n_samples = vis.shape[0]
        return (dweights / n_samples, 
                dbias_vis / n_samples, 
                dbias_hid / n_samples)

    def cd_update(self, vis):
        """Returns a tuple for the constrative divergence updates of
        the weights, visible biases and hidden biases."""
        model_vis, model_p_vis = self.gibbs_sample(vis, self.cd_steps)
        return self._cd_update_terms(vis, model_vis, model_p_vis)

    def pcd_update(self, vis):
        """Returns a tuple for the persistent constrative divergence updates of
        the weights, visible biases and hidden biases."""
        model_vis, model_p_vis = self.gibbs_sample(self.persistent_vis, 
                                                   self.cd_steps)
        self.persistent_vis = model_vis
        return self._cd_update_terms(vis, model_vis, model_p_vis)

    def reconstruction_cross_entropy(self, vis):
        """Returns the cross entropy between vis and its reconstruction 
        obtained by one step of Gibbs sampling."""
        _, sampled_p_vis = self.gibbs_sample(vis, 1)
        cross_entropy = gp.mean(vis * gp.log(sampled_p_vis) - 
                                (1 - vis) * gp.log(1-sampled_p_vis),
                                axis=1)
        return cross_entropy

    def pseudo_likelihood_for_bit(self, vis, i):
        """Returns the likelihood of bit i of vis given all other bits
        of vis."""
        fe = self.free_energy(vis)
        vis_flip = vis
        vis_flip[:,i] = 1 - vis[:,i]
        fe_flip = self.free_energy(vis_flip)
        pl = gp.log(gp.logistic(fe_flip - fe))
        return pl

    def pseudo_likelihood(self, vis):
        """Returns the pseudo likelihood of vis. 
        (summed over all samples in vis)"""
        pls = [gp.sum(self.pseudo_likelihood_for_bit(vis, i))
               for i in range(vis.shape[1])]
        return sum(pls)


    
def train_rbm(tcfg, print_cost=False):
    """Trains and returns an RBM using the specified 
    RestrictedBoltzmannMachineTrainingConfiguration"""
    # seed RNGs
    gp.seed_rand(tcfg.seed)

    # Build RBM
    rbm = RestrictedBoltzmannMachine(tcfg.batch_size, 
                                     tcfg.n_vis, 
                                     tcfg.n_hid, 
                                     tcfg.n_gibbs_steps, 
                                     tcfg.init_weight_sigma,
                                     tcfg.init_bias_sigma) 

    # initialize momentums
    weights_update = 0
    bias_vis_update = 0
    bias_hid_update = 0

    # train
    for epoch in range(tcfg.epochs):
        seen_epoch_samples = 0

        if print_cost:
            pl_bit = 0
            pl_sum = 0
            rc_sum = 0

        for x in draw_slices(tcfg.X, tcfg.batch_size, kind='sequential', 
                             samples_are='rows', stop=True):
            #print >>stderr, "%d / %d (epoch: %d / %d)\r" % (seen_epoch_samples, 
            #                                                tcfg.X.shape[0], 
            #                                                epoch, tcfg.epochs),

            # binaraize x
            if tcfg.binarize_data:
                x = sample_binomial(x)

            # perform weight update
            if tcfg.use_pcd:
                weights_step, bias_vis_step, bias_hid_step = rbm.pcd_update(x)
            else:
                weights_step, bias_vis_step, bias_hid_step = rbm.cd_update(x)

            if epoch >= tcfg.use_final_momentum_from_epoch:
                momentum = tcfg.final_momentum
            else:
                momentum = tcfg.initial_momentum
        
            weights_update = momentum * weights_update + \
                tcfg.step_rate * (weights_step - tcfg.weight_cost * rbm.weights)
            bias_vis_update = momentum * bias_vis_update + tcfg.step_rate * bias_vis_step
            bias_hid_update = momentum * bias_hid_update + tcfg.step_rate * bias_hid_step
    
            rbm.weights += weights_update
            rbm.bias_vis += bias_vis_update
            rbm.bias_hid += bias_hid_update

            seen_epoch_samples += tcfg.batch_size

            if print_cost:
                # calculate part of pseudo-likelihood
                pl_sum += gp.sum(rbm.pseudo_likelihood_for_bit(x > 0.5, pl_bit))
                pl_bit = (pl_bit + 1) % tcfg.X.shape[1]

                # calculate part of reconstruction cost
                rc_sum += gp.sum(rbm.reconstruction_cross_entropy(x > 0.5))

        #############################################
        # end of batch

        # save parameters
        save_parameters(rbm, epoch)

        # plot weights and current state of PCD chains
        plot_weights(rbm, epoch)
        if tcfg.use_pcd:
            plot_pcd_chains(rbm, epoch)

        if print_cost:
            # calculate pseudo likelihood and reconstruction cost
            pl = pl_sum / seen_epoch_samples * tcfg.X.shape[1]
            rc = rc_sum / seen_epoch_samples
            print "Epoch %02d: reconstruction cost=%f, pseudo likelihood=%f" % \
                (epoch, rc, pl)

    return rbm