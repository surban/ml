# -*- coding: utf-8 -*-

import gnumpy as gp
import numpy as np
import decimal
from sys import stderr
from time import time

import ml.common
import ml.common.util

from .util import sample_binomial, all_states, save_parameters, plot_weights, \
    plot_pcd_chains, load_parameters
from ml.common.util import logsum, draw_slices


class RestrictedBoltzmannMachine(object):
    "A Restricted Boltzmann Machine (RBM) with binary units"

    def __init__(self, batch_size, n_vis, n_hid, cd_steps):
        """A Restricted Boltzmann Machine (RBM) with binary units.
        batch_size is the size of a batch used for training. It may be 0 if
        the RBM will not be trained.
        n_vis and n_hid are the number of visible and hidden units.
        cd_steps is the number of alternating Gibbs sampling iterations to
        perform when computing the paramter updates using constrastive divergence."""
        self.weights = gp.zeros((n_vis, n_hid))
        self.bias_vis = gp.zeros((n_vis,))
        self.bias_hid = gp.zeros((n_hid,))

        self.cd_steps = cd_steps

        if batch_size > 0:
            self.persistent_vis = \
                gp.as_garray(np.random.normal(0, 1, size=(batch_size, n_vis)))
        
        self.log_pf = None
        self.log_pf_high = None
        self.log_pf_low = None

    def init_weights_zero(self):
        """Initis all weights to zero"""
        self.weights = gp.zeros((self.n_vis, self.n_hid))
        self.bias_vis = gp.zeros((self.n_vis,))
        self.bias_hid = gp.zeros((self.n_hid,))

    def init_weights_normal(self, weight_sigma, bias_sigma):
        """Inits the weights and biases with samples from a normal distribution
        with mean 0 and variance init_weight_sigma^2 / init_bias_sigma^2."""
        self.init_weights_zero()
        if weight_sigma > 0:
            self.weights = gp.as_garray(np.random.normal(0, weight_sigma, 
                                                         size=(self.n_vis, self.n_hid)))
        if bias_sigma > 0:
            self.bias_vis = gp.as_garray(np.random.normal(0, bias_sigma, 
                                                          size=(self.n_vis,)))
            self.bias_hid = gp.as_garray(np.random.normal(0, bias_sigma, 
                                                          size=(self.n_hid,)))

    def init_weights_uniform(self, weight_range, bias_range):
        """Inits the weights and biases with samples from a uniform distribution
        with support [-weight_range, weight_range] / [-bias_range, bias_range]."""
        self.init_weights_zero()
        self.weights = gp.as_garray(np.random.uniform(-weight_range, weight_range,
                                                      size=(self.n_vis, self.n_hid)))
        self.bias_vis = gp.as_garray(np.random.uniform(-bias_range, bias_range,
                                                       size=(self.n_vis,)))
        self.bias_hid = gp.as_garray(np.random.uniform(-bias_range, bias_range,
                                                       size=(self.n_hid,)))


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

    def free_energy(self, vis, beta=1):
        """The negative log probability of the visible units being in state vis 
        (without the normalization constant)"""
        return (- beta * gp.dot(vis, self.bias_vis) 
                - gp.sum(gp.log_1_plus_exp((self.bias_hid + gp.dot(vis, self.weights)) / beta),
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
            batches = ml.common.util.pack_in_batches(all_states(self.n_hid),
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

    def p_hid_given_vis(self, vis, beta=1):
        """Returns a vector whose ith component is the probability that the ith
        hidden unit is active given the states of the visible units"""
        return gp.logistic(beta * (gp.dot(vis, self.weights) + self.bias_hid))

    def sample_hid_given_vis(self, vis, beta=1):
        """Samples the hidden units given the visible units"""
        p = self.p_hid_given_vis(vis, beta)
        return sample_binomial(p)

    def p_vis_given_hid(self, hid, beta=1):
        """Returns a vector whose ith component is the probability that the ith
        visible unit is active given the states of the hidden units"""
        return gp.logistic(beta * (gp.dot(hid, self.weights.T) + self.bias_vis))

    def sample_vis_given_hid(self, hid, beta=1):
        """Samples the visible units given the hidden units"""
        p = self.p_vis_given_hid(hid, beta)
        return sample_binomial(p)

    def gibbs_sample(self, vis_start, k, beta=1, vis_force=None):
        """Performs k steps of alternating Gibbs sampling. Returns a tuple
        consisting of the final state of the visible units and the probability
        that they are active given the state of the hiddens in the previous
        to last step."""

        vis = vis_start
        for i in range(k):
            hid = self.sample_hid_given_vis(vis, beta)
            vis = self.sample_vis_given_hid(hid, beta)
            if vis_force is not None:
                ml.common.util.masked_set(vis, vis_force, vis_start)

        p_vis = self.p_vis_given_hid(hid, beta)
        if vis_force is not None:
            ml.common.util.masked_set(p_vis, vis_force, vis_start)

        return vis, p_vis

    def annealed_gibbs_sample(self, vis, betas):
        for beta in betas:
            beta = float(beta)
            hid = self.sample_hid_given_vis(vis, beta)
            vis = self.sample_vis_given_hid(hid, beta)
        p_vis = self.p_vis_given_hid(hid, float(betas[-1]))
        return vis, p_vis

    def free_energies_during_gibbs_sampling(self, x, kmax, beta=1):
        fes = []
        fes.append(gp.as_numpy_array(self.free_energy(x)))
        for k in range(kmax):
            x, _ = self.gibbs_sample(x, 1, beta=beta)
            fes.append(gp.as_numpy_array(self.free_energy(x)))
        fes=np.asarray(fes)
        return fes

    def strict_flip_sample(self, vis_start, iterations, beta=1):
        """Flips a randomly chosen bit and accepts the change if the
        resulting free energy is lower. Repeats for given iterations."""
        vis = vis_start.copy()
        fes = self.free_energy(vis) 
        n_total_flips = 0
    
        for i in range(iterations):
            # flip a bit at random
            f = np.random.randint(0, vis.shape[1])
            vis_prop = vis.copy()
            vis_prop[:,f] = 1-vis[:,f]
        
            # calculate new free energy and accept change if it is lower
            fes_prop = self.free_energy(vis_prop, beta=beta)
            acc_prop = fes_prop <= fes
            n_flips = gp.sum(acc_prop)
            n_total_flips += n_flips
        
            # compose new state
            acc_prop_t = gp.tile(acc_prop, (vis.shape[1], 1)).T
            vis = acc_prop_t * vis_prop + (1-acc_prop_t) * vis
            fes = acc_prop * fes_prop + (1-acc_prop) * fes
        
        return vis

    def metropolis_flip_sample(self, vis_start, iterations, beta=1, abeta=1):
        """Flips a randomly chosen bit and accepts the change if the
        resulting free energy is lower or with probability exp(-abeta*dE)
        where dE is the positive difference in energy. 
        Repeats for given iterations."""
        vis = vis_start.copy()
        fes = self.free_energy(vis)
        n_total_flips = 0
    
        for i in range(iterations):
            # flip a bit at random
            f = np.random.randint(0, vis.shape[1])
            vis_prop = vis.copy()
            vis_prop[:,f] = 1-vis[:,f]
        
            # calculate new free energy 
            fes_prop = self.free_energy(vis_prop, beta=beta)
            fes_diff = fes_prop - fes
        
            # accept if it is lower or with negative exponential probability
            fes_smaller = fes_diff <= 0
            acc_p = fes_smaller + (1-fes_smaller) * gp.exp(-(1-fes_smaller)*abeta*fes_diff)
            acc_rng = gp.rand(acc_p.shape)
            acc = acc_rng <= acc_p
        
            # statistics
            n_flips = gp.sum(acc)
            n_total_flips += n_flips
        
            # compose new state
            acc_t = gp.tile(acc, (vis.shape[1], 1)).T
            vis = acc_t * vis_prop + (1-acc_t) * vis
            fes = acc * fes_prop + (1-acc) * fes
        
        #print "Total number of flips: ", n_total_flips
        return vis

    def sample_vis(self, n_chains, n_steps, gibbs_steps_between_samples,
                   sample_probabilities=False, init_vis=None, beta=1):
        """Same as sample_vis_3d but concatenates all samples into a 2d array"""
        samples3d = self.sample_vis_3d(n_chains, n_steps, gibbs_steps_between_samples,
                                       sample_probabilities, init_vis, beta=beta)
        samples2d = gp.zeros((n_chains * n_steps, self.n_vis))
        for step in range(n_steps):
            samples2d[step*n_chains : (step+1)*n_chains, :] = \
                samples3d[step, :, :]
        return samples2d

    def sample_vis_3d(self, n_chains, n_steps, gibbs_steps_between_samples,
                      sample_probabilities=False, init_vis=None, beta=1,
                      betas=None):
        """Obtains unbiased samples for the visible units.
        Runs n_chains Gibbs chains in parallel for 
        (n_steps*gibbs_steps_between_samples) steps.
        Grabs samples every gibbs_steps_between_samples Gibbs steps."""
        samples = gp.zeros((n_steps, n_chains, self.n_vis))
        if init_vis is None:
            vis = gp.rand((n_chains, self.n_vis)) < 0.5
        else:
            assert init_vis.shape[0] == n_chains
            vis = init_vis

        for step in range(n_steps):
            #print >>stderr, "%d / %d                 \r" % (step, n_steps),
            if betas is None:
                vis, p_vis = self.gibbs_sample(vis, gibbs_steps_between_samples,
                                               beta=beta)
            else:
                assert gibbs_steps_between_samples is None
                vis, p_vis = self.annealed_gibbs_sample(vis, betas)
            if sample_probabilities:
                sample = p_vis
            else:
                sample = vis
            samples[step, :, :] = sample
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


    
def train_rbm(tcfg, print_cost=False, start_from_epoch=0):
    """Trains and returns an RBM using the specified 
    RestrictedBoltzmannMachineTrainingConfiguration"""

    start_time = time()

    # seed RNGs
    gp.seed_rand(tcfg.seed)

    # Build RBM
    rbm = RestrictedBoltzmannMachine(tcfg.batch_size, 
                                     tcfg.n_vis, 
                                     tcfg.n_hid, 
                                     tcfg.n_gibbs_steps)

    # init or load weights
    if start_from_epoch == 0:
        if tcfg.init_method == 'normal':
            ml.rbm.init_weights_normal(tcfg.init_weight_sigma, tcfg.init_bias_sigma)
        elif tcfg.init_method == 'uniform':
            ml.rbm.init_weights_uniform(tcfg.init_weight_sigma, tcfg.init_bias_sigma)
        else:
            assert False
    else:
        load_parameters(rbm, start_from_epoch - 1)

    # initialize momentums
    weights_update = 0
    bias_vis_update = 0
    bias_hid_update = 0

    # train
    for epoch in range(start_from_epoch, tcfg.epochs):
        seen_epoch_samples = 0

        if print_cost:
            pl_bit = 0
            pl_sum = 0
            rc_sum = 0

        if epoch >= tcfg.use_final_momentum_from_epoch:
            momentum = tcfg.final_momentum
        else:
            momentum = tcfg.initial_momentum

        for x in draw_slices(tcfg.X, tcfg.batch_size, kind='sequential', 
                             samples_are='rows', stop=True):
            #print >>stderr, "%d / %d (epoch: %d / %d)\r" % (seen_epoch_samples, 
            #                                                tcfg.X.shape[0], 
            #                                                epoch, tcfg.epochs),

            # binaraize x
            if tcfg.binarize_data == 'random':
                x = sample_binomial(x)

            # perform weight update
            if tcfg.use_pcd:
                weights_step, bias_vis_step, bias_hid_step = ml.rbm.pcd_update(x)
            else:
                weights_step, bias_vis_step, bias_hid_step = ml.rbm.cd_update(x)
       
            weights_update = momentum * weights_update + \
                tcfg.step_rate * (weights_step - tcfg.weight_cost * ml.rbm.weights)
            bias_vis_update = momentum * bias_vis_update + tcfg.step_rate * bias_vis_step
            bias_hid_update = momentum * bias_hid_update + tcfg.step_rate * bias_hid_step
    
            ml.rbm.weights += weights_update
            ml.rbm.bias_vis += bias_vis_update
            ml.rbm.bias_hid += bias_hid_update

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
        elif ml.common.show_progress:
            print "Epoch %02d completed" % epoch

    end_time = time()
    print "Training took %d s" % (end_time - start_time)

    return rbm