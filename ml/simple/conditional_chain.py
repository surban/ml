from __future__ import division

import numpy as np
from scipy.misc import logsumexp
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.mixture import GMM
from sklearn.neighbors.kde import KernelDensity

from ml.common.progress import status, done
from ml.simple.factorgraph import FactorGraph, Variable, Factor
from ml.simple.vargp import VarGP, normal_pdf


def find_evidence(state, state_next, train_states, input=None, train_inputs=None):
    val = (train_states == state) & (np.roll(train_states, -1, axis=0) == state_next)
    if input is not None:
        val &= (train_inputs == input)
    smpl_val = np.any(val, axis=0)
    return np.nonzero(smpl_val)[0]


def plot_transition(states, inputs, valid, pstate, style='x'):
    mask = np.roll((states == pstate) & valid, 1, axis=0) & valid
    if not np.any(mask):
        print "no"
        return

    i = inputs[mask]
    s = states[mask]
    i, s, _ = merge_duplicates(i, s)

    idx = np.argsort(i)
    i = i[idx]
    s = s[idx]

    plt.plot(i, s, style)


def plot_confidence(x, y, yerr, edgecolor, facecolor):
    pass


def normal_fit(x):
    # x[var, sample]
    assert x.ndim == 2

    # TODO: use robust covariance estimator sklearn.covariance.MinCovDet
    mu = np.mean(x, axis=1)
    sigma = np.cov(x)
    if x.shape[0] == 1:
        if sigma == 0:
            sigma = 0.01
        sigma = np.asarray([[sigma]])
    return mu, sigma


def merge_duplicates(x, y):
    assert x.ndim == 1 and y.ndim == 1

    xn = {}
    for i in range(x.size):
        k = x[i]
        if k in xn:
            xn[k].append(y[i])
        else:
            xn[k] = [y[i]]

    xout = np.zeros(len(xn.keys()), dtype='int')
    yout = np.zeros(xout.shape)
    yvar = np.zeros(xout.shape)
    for i, (k, v) in enumerate(xn.iteritems()):
        xout[i] = k
        yout[i] = np.mean(v)
        yvar[i] = np.var(v, ddof=1)

    return xout, yout, yvar


def iterate_over_samples(func, valid, *inps, **kwargs):
    show_progress = True
    if 'show_progress' in kwargs:
        show_progress = kwargs['show_progress']
        del kwargs['show_progress']

    assert valid.ndim == 2

    n_inps = len(inps)
    n_steps = valid.shape[0]
    n_samples = valid.shape[1]

    for inp in inps:
        assert inp.shape[-1] == n_samples

    outs = []

    for smpl in range(n_samples):
        if show_progress:
            status(smpl, n_samples)

        n_valid = np.where(valid[:, smpl])[0][-1] + 1

        smpl_inps = []
        for inp in inps:
            if inp.ndim == 1:
                smpl_inps.append(inp[smpl])
            else:
                smpl_inps.append(inp[..., 0:n_valid, smpl])
        smpl_inps = tuple(smpl_inps)

        smpl_outs = func(*smpl_inps, **kwargs)
        if not isinstance(smpl_outs, tuple):
            smpl_outs = (smpl_outs,)

        if smpl == 0:
            for smpl_out in smpl_outs:
                if isinstance(smpl_out, np.ndarray):
                    out = np.zeros(smpl_out.shape[0:-1] + (n_steps, n_samples), dtype=smpl_out.dtype)
                else:
                    out = np.zeros(n_samples, dtype=type(smpl_out))
                outs.append(out)

        for i, smpl_out in enumerate(smpl_outs):
            if isinstance(smpl_out, np.ndarray):
                outs[i][..., 0:n_valid, smpl] = smpl_out
            else:
                outs[i][smpl] = smpl_out

    if show_progress:
        done()

    if len(outs) == 1:
        return outs[0]
    else:
        return tuple(outs)


########################################################################################################################
########################################################################################################################
########################################################################################################################

class ConditionalChain(object):

    def __init__(self, n_system_states, n_input_values):
        # log_p_next_state[s_t, s_(t-1), f_t] = p(s_t | s_(t-1), f_t)
        self.log_p_next_state = np.zeros((n_system_states, n_system_states, n_input_values))

        # log_p_initial_state[s_0] = p(s_0)
        self.log_p_initial_state = np.zeros((n_system_states,))

    @property
    def n_system_states(self):
        return self.log_p_next_state.shape[0]

    @property
    def n_input_values(self):
        return self.log_p_next_state.shape[2]

    def train_table(self, states, inputs, valid):
        # states[step, sample]
        # inputs[step, sample]
        # valid[step, sample]

        n_steps = states.shape[0]
        n_samples = states.shape[1]
        assert np.all(0 <= states) and np.all(states < self.n_system_states)
        assert np.all(0 <= inputs) and np.all(inputs < self.n_input_values)
        assert states.dtype == 'int' and inputs.dtype == 'int'

        # initial state probabilities
        self.log_p_initial_state[:] = 0
        for smpl in range(n_samples):
            self.log_p_initial_state[states[0, smpl]] += 1
        self.log_p_initial_state /= n_samples
        self.log_p_initial_state = np.log(self.log_p_initial_state)

        # state transition probabilities
        self.log_p_next_state[:] = 0
        for step in range(1, n_steps):
            for smpl in range(n_samples):
                if not valid[step, smpl]:
                    continue
                self.log_p_next_state[states[step, smpl], states[step-1, smpl], inputs[step, smpl]] += 1

        # normalize
        sums = np.sum(self.log_p_next_state, axis=0)
        self.log_p_next_state /= sums[np.newaxis, :, :]
        self.log_p_next_state = np.log(self.log_p_next_state)

    def train_kde(self, states, inputs, valid, kernel, bandwidth):
        # states[step, sample]
        # inputs[step, sample]
        # valid[step, sample]

        n_steps = states.shape[0]
        n_samples = states.shape[1]
        assert np.all(0 <= states) and np.all(states < self.n_system_states)
        assert np.all(0 <= inputs) and np.all(inputs < self.n_input_values)
        assert states.dtype == 'int' and inputs.dtype == 'int'

        # initial state probabilities
        mu, sigma = normal_fit(states[0, :][np.newaxis, :])
        s = np.arange(self.n_system_states)
        self.log_p_initial_state[:] = normal_pdf(s[np.newaxis, :], mu, sigma)
        self.log_p_initial_state /= np.sum(self.log_p_initial_state)
        self.log_p_initial_state = np.log(self.log_p_initial_state)

        # state transition probabilities
        self.log_p_next_state[:] = 0
        for pstate in range(self.n_system_states):
            mask = np.roll((states == pstate) & valid, 1, axis=0) & valid
            if not np.any(mask):
                continue
            print pstate

            si = np.vstack((states[mask], inputs[mask]))

            kde = KernelDensity(kernel=kernel, bandwidth=bandwidth, rtol=1e-4)
            kde.fit(si.T)

            s, i = np.meshgrid(np.arange(self.n_system_states), np.arange(self.n_input_values))
            si = np.vstack((np.ravel(s.T), np.ravel(i.T)))
            pdf = np.exp(kde.score_samples(si.T))
            self.log_p_next_state[:, pstate, :] = np.reshape(pdf, (self.n_system_states, self.n_input_values))
            self.log_p_next_state[:, pstate, :] /= np.sum(self.log_p_next_state[:, pstate, :], axis=0)[np.newaxis, :]
        self.log_p_next_state = np.log(self.log_p_next_state)

    def get_transitions(self, states, inputs, valid, pstate):
        mask = np.roll((states == pstate) & valid, 1, axis=0) & valid
        return states[mask], inputs[mask]

    def plot_transitions(self, states, inputs, valid, pstate):
        s_raw, i_raw = self.get_transitions(states, inputs, valid, pstate)
        i, s, s_var = merge_duplicates(i_raw, s_raw)

        plt.subplot(1, 2, 1)
        plt.plot(i_raw, s_raw, '.')
        plt.axhline(pstate, color='r')
        plt.xlim(0, self.n_input_values)
        plt.ylim(0, self.n_system_states)
        plt.xlabel("input")
        plt.ylabel("next state")
        plt.title("raw transitions for state %d" % pstate)

        plt.subplot(1, 2, 2)
        plt.plot(i, s, '.')
        plt.errorbar(i, s, s_var, fmt=None)

        plt.axhline(pstate, color='r')
        plt.xlim(0, self.n_input_values)
        plt.ylim(0, self.n_system_states)
        plt.xlabel("input")
        plt.ylabel("next state")
        plt.title("mean transitions for state %d" % pstate)

    def train_gp_var(self, states, inputs, valid, mngful_dist=None, plot_pstate=None, **kwargs):
        # states[step, sample]
        # inputs[step, sample]
        # valid[step, sample]

        assert np.all(0 <= states) and np.all(states < self.n_system_states)
        assert np.all(0 <= inputs) and np.all(inputs < self.n_input_values)
        assert states.dtype == 'int' and inputs.dtype == 'int'

        # initial state probabilities
        mu, sigma = normal_fit(states[0, :][np.newaxis, :])
        s = np.arange(self.n_system_states)
        self.log_p_initial_state[:] = normal_pdf(s[np.newaxis, :], mu, sigma)
        self.log_p_initial_state /= np.sum(self.log_p_initial_state)
        self.log_p_initial_state = np.log(self.log_p_initial_state)

        # state transition probabilities
        self.log_p_next_state[:] = 0
        for pstate in range(self.n_system_states):
            if plot_pstate is not None and plot_pstate != pstate:
                continue
            status(pstate, self.n_system_states, "Training GP")
            # print pstate

            sta, inp = self.get_transitions(states, inputs, valid, pstate)
            if sta.size == 0:
                continue
            inp = inp[np.newaxis, :]
            vargp = VarGP(inp, sta, **kwargs)
            if mngful_dist is not None:
                vargp.limit_meaningful_predictions(mngful_dist, inp, self.n_input_values)
            self.log_p_next_state[:, pstate, :] = vargp.pdf_for_all_inputs(self.n_input_values, self.n_system_states)
            if plot_pstate is not None:
                vargp.plot(inp, sta, self.n_input_values, self.n_input_values, hmarker=pstate)
                return
        done()

        self.log_p_next_state = np.log(self.log_p_next_state)

    def check_normalization(self):
        # self.log_p_initial_state[s_0] = p(s_0)
        # self.log_p_next_state[s_t, s_(t-1), f_t] = p(s_t | s_(t-1), f_t)

        is_sum = logsumexp(self.log_p_initial_state)
        ns_sum = logsumexp(self.log_p_next_state, axis=0)

        print "Initial state: ", is_sum
        print "Next state normalization failures: "

        for pstate in range(self.n_system_states):
            for inp in range(self.n_input_values):
                if not np.all(np.isfinite(self.log_p_next_state[:, pstate, inp])):
                    ns_sum[pstate, inp] = 0

        err_pos = np.where(np.abs(ns_sum) > 1e-5)
        for pstate, inp in np.transpose(err_pos):
            print "pstate=%d inp=%d:   %f" % (pstate, inp, ns_sum[pstate, inp])

    def infer_inputs(self, states):
        # states[step]
        # ml_inputs[step]
        # log_p_inputs[input, step]
        # self.log_p_initial_state[s_0] = p(s_0)
        # self.log_p_next_state[s_t, s_(t-1), f_t] = p(s_t | s_(t-1), f_t)

        n_steps = states.shape[0]
        assert np.all(0 <= states) and np.all(states < self.n_system_states)

        # initialize outputs
        log_p_inputs = np.zeros((self.n_input_values, n_steps))

        # initial state
        pstate = np.argmax(self.log_p_initial_state)

        # infer input probabilities using Bayes' theorem
        for step in range(n_steps):
            state = states[step]
            log_p_inputs[:, step] = (self.log_p_next_state[state, pstate, :] -
                                     logsumexp(self.log_p_next_state[state, pstate, :]))
            pstate = state

        # determine maximum probability inputs
        ml_inputs = np.argmax(log_p_inputs, axis=0)

        return ml_inputs, log_p_inputs

    def most_probable_inputs(self, states, sigma):
        # states[step]
        # msg[f_(step-1)]
        # max_track[step, f_step]
        # max_input[step]
        # p_next_input[f_step, f_(step-1)]
        # self.log_p_next_state[s_t, s_(t-1), f_t] = p(s_t | s_(t-1), f_t)

        n_steps = states.shape[0]
        assert np.all(0 <= states) and np.all(states < self.n_system_states)

        # precompute input transition probabilities
        all_inputs = np.arange(0, self.n_input_values)
        p_next_input = np.zeros((self.n_input_values, self.n_input_values))
        for pinp in range(self.n_input_values):
            p_next_input[:, pinp] = scipy.stats.norm.pdf(all_inputs, loc=pinp, scale=sigma)
            p_next_input[:, pinp] /= np.sum(p_next_input[:, pinp])
        log_p_next_input = np.log(p_next_input)

        # track maximum probability inputs
        max_track = np.zeros((n_steps, self.n_input_values), dtype='int')

        # use uniform probability for leaf factor at beginning of chain
        initial_prob = np.ones(self.n_input_values) / float(self.n_input_values)
        msg = np.log(initial_prob)

        # initial state
        pstate = np.argmax(self.log_p_initial_state)

        # pass messages towards end node of chain
        for step in range(n_steps):
            # m[f_t, f_(t-1)]
            state = states[step]
            log_p_next_state = self.log_p_next_state[state, pstate, :, np.newaxis]
            if not np.any(np.isfinite(log_p_next_state)):
                log_p_next_state = np.log(np.ones(log_p_next_state.shape) / float(log_p_next_state.shape[0]))
            m = log_p_next_state + log_p_next_input + msg[np.newaxis, :]
            max_track[step, :] = np.nanargmax(m, axis=1)
            msg = np.nanmax(m, axis=1)
            pstate = state

        # calculate maximum probability
        log_p_max = np.nanmax(msg)

        # backtrack maximum states
        max_input = np.zeros(n_steps, dtype='int')
        max_input[n_steps-1] = np.nanargmax(msg)
        for step in range(n_steps-2, -1, -1):
            max_input[step] = max_track[step+1, max_input[step+1]]

        return max_input, log_p_max

    def most_probable_states(self, inputs):
        # inputs[step]
        # msg[s_(step-1)]
        # max_track[step, s_step]
        # max_state[step]

        n_steps = inputs.shape[0]
        assert np.all(0 <= inputs) and np.all(inputs < self.n_input_values)

        # track maximum states
        max_track = np.zeros((n_steps, self.n_system_states), dtype='int')

        # leaf factor at beginning of chain
        msg = self.log_p_initial_state

        # pass messages towards end node of chain
        for step in range(n_steps):
            # m[s_t, s_(t-1)]
            m = self.log_p_next_state[:, :, inputs[step]] + msg[np.newaxis, :]
            max_track[step, :] = np.nanargmax(m, axis=1)
            msg = np.nanmax(m, axis=1)

        # calculate maximum probability
        log_p_max = np.nanmax(msg)

        # backtrack maximum states
        max_state = np.zeros(n_steps, dtype='int')
        max_state[n_steps-1] = np.nanargmax(msg)
        for step in range(n_steps-2, -1, -1):
            max_state[step] = max_track[step+1, max_state[step+1]]

        return max_state, log_p_max

    def log_p(self, states, inputs):
        n_steps = inputs.shape[0]
        assert np.all(0 <= states) and np.all(states < self.n_system_states)
        assert np.all(0 <= inputs) and np.all(inputs < self.n_input_values)

        lp = self.log_p_initial_state[states[0]]
        for step in range(1, n_steps):
            lp += self.log_p_next_state[states[step], states[step-1], inputs[step]]
        return lp

    def greedy_states(self, inputs):
        n_steps = inputs.shape[0]
        assert np.all(0 <= inputs) and np.all(inputs < self.n_input_values)

        max_state = np.zeros(n_steps, dtype='int')
        cur_state = np.argmin(self.log_p_initial_state)
        for step in range(0, n_steps):
            # print "step=%d, state=%d, input=%d, p(s_t+1 | state, input)=" % (step, cur_state, inputs[step])
            # print self.log_p_next_state[:, cur_state, inputs[step]]

            cur_state = np.argmax(self.log_p_next_state[:, cur_state, inputs[step]])
            max_state[step] = cur_state

        return max_state

    def greedy_next_state(self, states, inputs):
        n_steps = inputs.shape[0]
        assert np.all(0 <= states) and np.all(states < self.n_system_states)
        assert np.all(0 <= inputs) and np.all(inputs < self.n_input_values)

        max_state = np.zeros(n_steps, dtype='int')
        max_state[0] = states[0]
        for step in range(1, n_steps):
            # print "step=%d, state=%d, input=%d, p(s_t+1 | state, input)=" % (step, states[step-1], inputs[step])
            # print self.log_p_next_state[:, states[step-1], inputs[step]]

            max_state[step] = np.argmax(self.log_p_next_state[:, states[step-1], inputs[step]])

        return max_state

########################################################################################################################
########################################################################################################################
########################################################################################################################


class ControlObservationChain(object):

    def __init__(self, n_system_states, n_input_values):
        # log_p_next_state[s_t, s_(t-1), f_t] = p(s_t | s_(t-1), f_t)
        self.log_p_next_state = np.zeros((n_system_states, n_system_states, n_input_values))

        # log_p_initial_state[s_0] = p(s_0)
        self.log_p_initial_state = np.zeros((n_system_states,))

        # GP that models p(s_t | x_t)
        self.gp_states_given_observations = None
        """:type : VarGP"""

        # GP that models p(s_t | s_(t-1), f_t)
        self.gp_all_transitions = None
        """:type : VarGP"""

        # GP that models p(f_t | x_t)
        self.gp_inputs_given_observations = None
        """:type : VarGP"""

        # GPs that model p(s_t | f_t) for each s_(t-1)
        self.gp_transitions = []

        # uniform prior over inputs
        self.log_p_input = np.log(np.ones(self.n_input_values) / self.n_input_values)

        # uniform prior over states
        self.log_p_state = np.log(np.ones(self.n_system_states) / self.n_system_states)

    @property
    def n_system_states(self):
        return self.log_p_next_state.shape[0]

    @property
    def n_input_values(self):
        return self.log_p_next_state.shape[2]

    def get_transitions(self, states, inputs, valid, pstate):
        mask = np.roll((states == pstate) & valid, 1, axis=0) & valid
        return states[mask], inputs[mask]

    def get_all_transitions(self, states, inputs, valid):
        pstates = np.roll(states, 1, axis=0)
        pvalid = np.roll(valid, 1, axis=0)
        mask = valid & pvalid
        return states[mask], pstates[mask], inputs[mask]

    def get_valid_observations(self, states, observations, valid):
        # states[step, sample]
        # observations[feature, step, sample]
        # valid[step, sample]

        n_samples = states.shape[1]
        n_features = observations.shape[0]

        valid_states = np.zeros(0)
        valid_observations = np.zeros((n_features, 0))

        for smpl in range(n_samples):
            n_valid_steps = np.where(valid[:, smpl])[0][-1] + 1
            valid_states = np.concatenate((valid_states, states[0:n_valid_steps, smpl]))
            valid_observations = np.concatenate((valid_observations, observations[:, 0:n_valid_steps, smpl]), axis=1)

        return valid_states, valid_observations

    def initialize_transitions(self, sigma):
        self.log_p_initial_state[:] = np.log(np.ones(self.n_system_states) / float(self.n_system_states))

        sigma_mat = np.asarray([[sigma]])
        states = np.arange(self.n_system_states)
        for pstate in range(self.n_system_states):
            for inp in range(self.n_input_values):
                mu = np.asarray([inp / float(self.n_input_values) * float(self.n_system_states)])
                self.log_p_next_state[:, pstate, inp] = normal_pdf(states[np.newaxis, :], mu, sigma_mat)
                self.log_p_next_state[:, pstate, inp] /= np.sum(self.log_p_next_state[:, pstate, inp])
        self.log_p_next_state = np.log(self.log_p_next_state)

    def train_transitions(self, states, inputs, valid, mngful_dist=None, plot_pstate=None,
                          finetune=False, finetune_optimize=False, **kwargs):
        # states[step, sample]
        # inputs[step, sample]
        # valid[step, sample]

        assert np.all(0 <= states) and np.all(states < self.n_system_states)
        assert np.all(0 <= inputs) and np.all(inputs < self.n_input_values)
        assert states.dtype == 'int' and inputs.dtype == 'int'

        # initial state probabilities
        mu, sigma = normal_fit(states[0, :][np.newaxis, :])
        s = np.arange(self.n_system_states)
        self.log_p_initial_state[:] = normal_pdf(s[np.newaxis, :], mu, sigma)
        self.log_p_initial_state /= np.sum(self.log_p_initial_state)
        self.log_p_initial_state = np.log(self.log_p_initial_state)

        if not finetune:
            self.gp_transitions = [None for _ in range(self.n_system_states)]

        # state transition probabilities
        self.log_p_next_state[:] = 0
        for pstate in range(self.n_system_states):
            if plot_pstate is not None and plot_pstate != pstate:
                continue
            status(pstate, self.n_system_states, "Learning transitions")

            sta, inp = self.get_transitions(states, inputs, valid, pstate)
            if sta.size == 0:
                continue
            inp = inp[np.newaxis, :]
            if finetune and self.gp_transitions[pstate] is not None:
                self.gp_transitions[pstate].train(inp, sta, optimize=finetune_optimize)
            else:
                self.gp_transitions[pstate] = VarGP(inp, sta, **kwargs)
            if mngful_dist is not None:
                self.gp_transitions[pstate].limit_meaningful_predictions(mngful_dist, inp, self.n_input_values)
            self.log_p_next_state[:, pstate, :] = \
                self.gp_transitions[pstate].pdf_for_all_inputs(self.n_input_values, self.n_system_states)
            if plot_pstate is not None:
                self.gp_transitions[pstate].plot(inp, sta, self.n_input_values, self.n_input_values, hmarker=pstate)
                return
        done()

        self.log_p_next_state = np.log(self.log_p_next_state)

    def train_all_transitions(self, states, inputs, valid, mngful_dist=None, **kwargs):
        # states[step, sample]
        # inputs[step, sample]
        # valid[step, sample]
        # self.log_p_next_state[s_t, s_(t-1), f_t] = p(s_t | s_(t-1), f_t)

        assert np.all(0 <= states) and np.all(states < self.n_system_states)
        assert np.all(0 <= inputs) and np.all(inputs < self.n_input_values)
        assert states.dtype == 'int' and inputs.dtype == 'int'

        # initial state probabilities
        mu, sigma = normal_fit(states[0, :][np.newaxis, :])
        s = np.arange(self.n_system_states)
        self.log_p_initial_state[:] = normal_pdf(s[np.newaxis, :], mu, sigma)
        self.log_p_initial_state /= np.sum(self.log_p_initial_state)
        self.log_p_initial_state = np.log(self.log_p_initial_state)

        # get all transitions and build training matrix
        sta, psta, inp = self.get_all_transitions(states, inputs, valid)
        psta_inp = np.vstack((psta, inp))

        # train GP model
        self.gp_all_transitions = VarGP(psta_inp, sta, **kwargs)
        if mngful_dist is not None:
            self.gp_all_transitions.limit_meaningful_predictions(mngful_dist, psta_inp, self.n_input_values)

        # calculate resulting PDF
        self.log_p_next_state = \
            self.gp_all_transitions.pdf_for_all_inputs((self.n_system_states, self.n_input_values),
                                                       self.n_system_states)
        self.log_p_next_state = np.log(self.log_p_next_state)

    def train_observations(self, states, observations, valid, finetune=False, finetune_optimize=False, **kwargs):
        # states[step, sample]
        # observations[feature, step, sample]
        # valid[step, sample]

        assert np.all(0 <= states) and np.all(states < self.n_system_states)
        assert states.dtype == 'int'

        v_states, v_observations = self.get_valid_observations(states, observations, valid)
        if finetune:
            self.gp_states_given_observations.train(v_observations, v_states, optimize=finetune_optimize)
        else:
            self.gp_states_given_observations = VarGP(v_observations, v_states, use_data_variance=False, **kwargs)

    def train_inputs_given_observations(self, states, observations, valid,
                                        finetune=False, finetune_optimize=False, **kwargs):
        # states[step, sample]
        # observations[feature, step, sample]
        # valid[step, sample]

        assert np.all(0 <= states) and np.all(states < self.n_system_states)
        assert states.dtype == 'int'

        v_states, v_observations = self.get_valid_observations(states, observations, valid)
        if finetune:
            self.gp_inputs_given_observations.train(v_observations, v_states, optimize=finetune_optimize)
        else:
            self.gp_inputs_given_observations = VarGP(v_observations, v_states, use_data_variance=False, **kwargs)

    def infer_states(self, observations):
        # observations[feature, step]
        # states[step]
        # states_std[step]

        states, states_std, _, _ = self.gp_states_given_observations.predict(observations)
        return states, states_std

    def infer_inputs(self, observations):
        # observations[feature, step]
        # states[step]
        # states_std[step]

        states, states_std, _, _ = self.gp_inputs_given_observations.predict(observations)
        return states, states_std

    def check_normalization(self):
        # self.log_p_initial_state[s_0] = p(s_0)
        # self.log_p_next_state[s_t, s_(t-1), f_t] = p(s_t | s_(t-1), f_t)

        is_sum = logsumexp(self.log_p_initial_state)
        ns_sum = logsumexp(self.log_p_next_state, axis=0)

        if not np.isclose(is_sum, 0):
            print "Initial state: ", is_sum

        for pstate in range(self.n_system_states):
            for inp in range(self.n_input_values):
                if not np.all(np.isfinite(self.log_p_next_state[:, pstate, inp])):
                    ns_sum[pstate, inp] = 0

        err_pos = np.where(np.abs(ns_sum) > 1e-5)
        if err_pos[0].size > 0:
            print "Next state normalization failures: "
        for pstate, inp in np.transpose(err_pos):
            print "pstate=%d inp=%d:   %f" % (pstate, inp, ns_sum[pstate, inp])

    def build_direct_factorgraph(self, observation_values, observation_to_input):
        # self.log_p_input[f_t] = p(f_t)
        # self.log_p_state[s_t] = p(s_t)
        # self.log_p_initial_state[s_-1] = p(s_-1)
        # self.log_p_next_state[s_t, s_(t-1), f_t] = p(s_t | s_(t-1), f_t)
        # log_p_input_given_obs
        # observation_values[feature, step]
        # log_p_state_given_obs[state, step]
        # log_p_obs[state, step]
        # log_p_next_input[f_(t+1), f_t]

        n_steps = observation_values.shape[1]

        # p(s_t | x_t)
        log_p_state_given_obs = np.log(self.gp_states_given_observations.pdf(observation_values, self.n_system_states))

        # p(f_t | x_t)
        log_p_input_given_obs = np.log(self.gp_inputs_given_observations.pdf(observation_values, self.n_input_values))

        # p(x_t) (dummy prior)
        log_p_obs = np.zeros(1)

        fg = FactorGraph()
        inputs = []
        input_factors = []
        states = []
        observations = []
        observations2 = []

        # initial state
        state_initial = Variable("s_initial", self.n_system_states)
        fg.add_node(state_initial)
        fg.add_node(Factor("S_INITIAL", self.log_p_initial_state, (state_initial,)))
        previous_state = state_initial

        # build graph parts for each timestep
        for step in range(n_steps):
            if observation_to_input:
                # observation (fixed)
                obs = Variable("x_%d" % step, 1)
                fg.add_node(obs)
                fg.add_node(Factor("X_%d" % step, log_p_obs, (obs,)))

            # observation 2 (fixed)
            obs2 = Variable("x'_%d" % step, 1)
            fg.add_node(obs2)
            fg.add_node(Factor("X'_%d" % step, log_p_obs, (obs2,)))

            # input
            inp = Variable("f_%d" % step, self.n_input_values)
            fg.add_node(inp)
            if observation_to_input:
                input_factor = Factor("F_%d" % step, log_p_input_given_obs[np.newaxis, :, step], (obs, inp))
            else:
                input_factor = Factor("F_%d" % step, self.log_p_input, (inp,))
            fg.add_node(input_factor)

            # state: p(s_t | s_(t-1), x_t, f_t)
            # log_p_state_given_all[s_t, s_(t-1), f_t, x_t]
            log_p_state_given_all = (log_p_state_given_obs[:, step, np.newaxis, np.newaxis, np.newaxis] +
                                     self.log_p_next_state[:, :, :, np.newaxis])
            log_p_state_given_all -= logsumexp(log_p_state_given_all, axis=0)
            log_p_state_given_all[np.isnan(log_p_state_given_all)] = -np.inf
            assert not np.any(np.isnan(log_p_state_given_all))

            state = Variable("s_%d" % step, self.n_system_states)
            fg.add_node(state)
            fg.add_node(Factor("S_%d" % step, log_p_state_given_all,
                               (state, previous_state, inp, obs2)))

            inputs.append(inp)
            input_factors.append(input_factor)
            states.append(state)
            if observation_to_input:
                observations.append(obs)
            observations2.append(obs2)
            previous_state = state

        return fg, inputs, input_factors, states, observations, observations2

    def direct_fg_most_probable_states_and_inputs_given_observations(self, observation_values, observation_to_input):
        fg, inputs, input_factors, states, observations, observations2 = \
            self.build_direct_factorgraph(observation_values, observation_to_input=observation_to_input)

        fg.prepare()
        fg.do_message_passing()

        best_log_p = fg.backtrack_best_state()
        best_state = np.asarray([s.best_state for s in states])
        best_input = np.asarray([i.best_state for i in inputs])

        return best_state, best_input, best_log_p

    def build_factorgraph(self, n_steps=None, observation_values=None, observation_for_inputs_values=None,
                          smooth_input_sigma=None):
        # self.log_p_input[f_t] = p(f_t)
        # self.log_p_state[s_t] = p(s_t)
        # self.log_p_initial_state[s_-1] = p(s_-1)
        # self.log_p_next_state[s_t, s_(t-1), f_t] = p(s_t | s_(t-1), f_t)
        # observation_values[feature, step]
        # observation_for_input_values[feature, step]
        # log_p_state_given_obs[state, step]
        # log_p_obs_given_state[state, step]
        # log_p_next_input[f_(t+1), f_t]
        # log_p_input_given_obs[state, step]
        # log_p_obs_given_input[state, step]

        if observation_values is not None:
            n_steps = observation_values.shape[1]
        if observation_for_inputs_values is not None:
            assert n_steps == observation_for_inputs_values.shape[1]
        assert n_steps is not None

        if observation_values is not None:
            # p(s_t | x_t)
            log_p_state_given_obs = np.log(self.gp_states_given_observations.pdf(observation_values,
                                                                                 self.n_system_states))

            # p(x_t | s_t) + const.
            log_p_obs_given_state = log_p_state_given_obs - self.log_p_state[:, np.newaxis]

        if observation_for_inputs_values is not None:
            # p(f_t | x_t)
            log_p_input_given_obs = np.log(self.gp_inputs_given_observations.pdf(observation_for_inputs_values,
                                                                                 self.n_input_values))

            # p(x'_t | f_t) + const.
            log_p_obs_given_input = log_p_input_given_obs - self.log_p_input[:, np.newaxis]

        # p(f_(t+1) | f_t)
        if smooth_input_sigma is not None:
            all_inputs = np.arange(0, self.n_input_values)
            p_next_input = np.zeros((self.n_input_values, self.n_input_values))
            for pinp in range(self.n_input_values):
                p_next_input[:, pinp] = scipy.stats.norm.pdf(all_inputs, loc=pinp, scale=smooth_input_sigma)
                p_next_input[:, pinp] /= np.sum(p_next_input[:, pinp])
            log_p_next_input = np.log(p_next_input)

        fg = FactorGraph()
        inputs = []
        input_factors = []
        states = []
        observations = []
        observations_for_inputs = []

        # initial state
        state_initial = Variable("s_initial", self.n_system_states)
        fg.add_node(state_initial)
        fg.add_node(Factor("S_INITIAL", self.log_p_initial_state, (state_initial,)))
        previous_state = state_initial

        # initial inputs
        if smooth_input_sigma is not None:
            input_initial = Variable("f_initial", self.n_input_values)
            fg.add_node(input_initial)
            fg.add_node(Factor("F_INITIAL", self.log_p_input, (input_initial,)))
            previous_inp = input_initial

        # build graph parts for each timestep
        for step in range(n_steps):
            # input
            inp = Variable("f_%d" % step, self.n_input_values)
            fg.add_node(inp)
            if smooth_input_sigma is not None:
                input_factor = Factor("F_%d" % step, log_p_next_input, (inp, previous_inp))
            else:
                input_factor = Factor("F_%d" % step, self.log_p_input, (inp,))
            fg.add_node(input_factor)

            # state
            state = Variable("s_%d" % step, self.n_system_states)
            fg.add_node(state)
            fg.add_node(Factor("S_%d" % step, self.log_p_next_state,
                               (state, previous_state, inp)))

            if observation_values is not None:
                # observation (fixed)
                obs = Variable("x_%d" % step, 1)
                fg.add_node(obs)
                fg.add_node(Factor("X_%d" % step, log_p_obs_given_state[np.newaxis, :, step], (obs, state)))

            if observation_for_inputs_values is not None:
                # observation for input (fixed)
                obs_for_inp = Variable("x'_%d" % step, 1)
                fg.add_node(obs_for_inp)
                fg.add_node(Factor("X'_%d" % step, log_p_obs_given_input[np.newaxis, :, step], (obs_for_inp, inp)))

            inputs.append(inp)
            input_factors.append(input_factor)
            states.append(state)
            if observation_values is not None:
                observations.append(obs)
            if observation_for_inputs_values is not None:
                observations_for_inputs.append(obs_for_inp)
            previous_state = state
            if smooth_input_sigma is not None:
                previous_inp = inp

        return fg, inputs, input_factors, states, observations, observations_for_inputs

    def fg_most_probable_states_and_inputs_given_observations(self, observation_values,
                                                              observation_for_inputs_values=None,
                                                              smooth_input=None, loopy_iters=None,
                                                              prepass_non_loopy=True):
        n_steps = observation_values.shape[1]

        fg, inputs, input_factors, states, observations, observations_for_inputs = \
            self.build_factorgraph(observation_values=observation_values,
                                   observation_for_inputs_values=observation_for_inputs_values,
                                   smooth_input_sigma=smooth_input)

        if loopy_iters is not None:
            if prepass_non_loopy:
                fg.loopy_propagation = False
                fg.prepare()

                if smooth_input is not None:
                    # inject uniform messages to break initial loops
                    for step in range(n_steps):
                        inputs[step].assume_uniform_msg(input_factors[step])
                        if step != n_steps - 1:
                            inputs[step].assume_uniform_msg(input_factors[step + 1])

                # do non-loopy propagation to precalculate probability estimates  
                fg.do_message_passing(only_leafs=False)

                # switch to loopy propagation to get smooth force estimates
                fg.loopy_propagation = True
                fg.do_message_passing(loopy_iters)
            else:
                fg.loopy_propagation = True
                fg.prepare()
                fg.do_message_passing(loopy_iters)
        else:
            fg.prepare()
            fg.do_message_passing()

        best_log_p = fg.backtrack_best_state()
        best_state = np.asarray([s.best_state for s in states])
        best_input = np.asarray([i.best_state for i in inputs])

        return best_state, best_input, best_log_p

    def fg_most_probable_states_given_inputs(self, input_values):
        n_steps = input_values.shape[0]
        fg, inputs, input_factors, states, observations, observations_for_inputs = \
            self.build_factorgraph(n_steps=n_steps)

        # clamp inputs
        for step in range(n_steps):
            inputs[step].clamp(input_values[step])

        fg.prepare()
        fg.do_message_passing()
        best_log_p = fg.backtrack_best_state()
        best_state = np.asarray([s.best_state for s in states])
        return best_state, best_log_p

    def _nanmax(self, x):
        xflat = np.reshape(x, (x.shape[0], -1))
        xflat_argmax = np.nanargmax(xflat, axis=1)
        x_max = np.nanmax(xflat, axis=1)
        x_argmax = np.unravel_index(xflat_argmax, x.shape[1:])
        return x_max, x_argmax

    def most_probable_states_and_inputs_given_observations(self, observations):
        # observations[feature, step]
        # log_p_state_given_obs[state, step]
        # obs_state_mean[step]
        # obs_state_std[step]
        # inputs[step]
        # log_p_input[input]
        # log_p_state[state]
        # msg[s_(step-1)]
        # mt_state[step, s_step]
        # mt_input[step, s_step]
        # best_state[step]
        # best_input[step]
        # self.log_p_next_state[s_t, s_(t-1), f_t] = p(s_t | s_(t-1), f_t)

        n_steps = observations.shape[1]

        # track maximum states and inputs
        mt_state = np.zeros((n_steps, self.n_system_states), dtype='int')
        mt_input = np.zeros((n_steps, self.n_system_states), dtype='int')

        # state probabilities from observations
        log_p_state_given_obs = np.log(self.gp_states_given_observations.pdf(observations, self.n_system_states))

        # leaf factor at beginning of chain
        msg = self.log_p_initial_state

        # pass messages towards end node of chain
        for step in range(n_steps):
            # m[s_t, s_(t-1), f_t]
            # m_max[s_t]
            # m_argmax_pstate[s_t]
            # m_argmax_input[s_t]
            m = self.log_p_next_state + self.log_p_input + msg[np.newaxis, :, np.newaxis]
            m_max, (m_argmax_pstate, m_argmax_input) = self._nanmax(m)
            mt_state[step, :] = m_argmax_pstate
            mt_input[step, :] = m_argmax_input
            msg = m_max + log_p_state_given_obs[:, step] - self.log_p_state

        # calculate maximum probability
        best_log_p = np.nanmax(msg)

        # backtrack best states and inputs
        best_state = np.zeros(n_steps, dtype='int')
        best_state[n_steps-1] = np.nanargmax(msg)
        for step in range(n_steps-2, -1, -1):
            best_state[step] = mt_state[step+1, best_state[step+1]]
        best_input = mt_input[np.arange(mt_input.shape[0]), best_state]

        return best_state, best_input, best_log_p

    def most_probable_states_given_inputs(self, inputs):
        return self.most_probable_states_given_inputs_and_observations(inputs, None)

    def most_probable_states_given_inputs_and_observations(self, inputs, observations):
        # inputs[step]
        # observations
        # msg[s_(step-1)]
        # mt_state[step, s_step]
        # best_state[step]
        # log_p_state_given_obs[state, step]

        n_steps = inputs.shape[0]
        assert inputs.dtype == 'int'
        assert np.all(0 <= inputs) and np.all(inputs < self.n_input_values)

        # track maximum states
        mt_state = np.zeros((n_steps, self.n_system_states), dtype='int')

        # state probabilities from observations
        if observations is not None:
            log_p_state_given_obs = np.log(self.gp_states_given_observations.pdf(observations, self.n_system_states))

        # leaf factor at beginning of chain
        msg = self.log_p_initial_state

        # pass messages towards end node of chain
        for step in range(n_steps):
            # m[s_t, s_(t-1)]
            m = self.log_p_next_state[:, :, inputs[step]] + msg[np.newaxis, :]
            m_max = np.nanmax(m, axis=1)
            mt_state[step, :] = np.nanargmax(m, axis=1)
            msg = m_max
            if observations is not None:
                msg += log_p_state_given_obs[:, step] - self.log_p_state

        # calculate maximum probability
        best_log_p = np.nanmax(msg)

        # backtrack maximum states
        best_state = np.zeros(n_steps, dtype='int')
        best_state[n_steps-1] = np.nanargmax(msg)
        for step in range(n_steps-2, -1, -1):
            best_state[step] = mt_state[step+1, best_state[step+1]]

        return best_state, best_log_p

    def log_p(self, states, inputs, observations):
        n_steps = inputs.shape[0]
        assert np.all(0 <= states) and np.all(states < self.n_system_states)
        assert np.all(0 <= inputs) and np.all(inputs < self.n_input_values)

        # state probabilities from observations
        log_p_state_given_obs = np.log(self.gp_states_given_observations.pdf(observations, self.n_system_states))

        lp = self.log_p_initial_state[states[0]]
        for step in range(1, n_steps):
            lp += self.log_p_next_state[states[step], states[step-1], inputs[step]]
            lp += log_p_state_given_obs[states[step], step] - self.log_p_state[states[step]]
        return lp


########################################################################################################################
########################################################################################################################
########################################################################################################################


class ConditionalFilteredChain(object):

    def __init__(self, n_system_states, n_input_values, input_std):
        # log_p_next_state[s_t, s_(t-1), f_t] = p(s_t, f_t | s_(t-1))
        self.p_next_state = np.zeros((n_system_states, n_system_states, n_input_values))

        # log_p_initial_state[s_0] = p(s_0)
        self.p_initial_state = np.zeros((n_system_states,))

        self.input_std = input_std

    @property
    def n_system_states(self):
        return self.p_next_state.shape[0]

    @property
    def n_input_values(self):
        return self.p_next_state.shape[2]

    def train(self, states, inputs, valid):
        # states[step, sample]
        # inputs[step, sample]
        # valid[step, sample]

        n_steps = states.shape[0]
        n_samples = states.shape[1]
        assert np.all(0 <= states) and np.all(states < self.n_system_states)
        assert np.all(0 <= inputs) and np.all(inputs < self.n_input_values)
        assert states.dtype == 'int' and inputs.dtype == 'int'

        # initial state probabilities
        self.p_initial_state[:] = 0
        for smpl in range(n_samples):
            self.p_initial_state[states[0, smpl]] += 1
        self.p_initial_state /= n_samples

        # state transition probabilities
        self.p_next_state[:] = 0
        for step in range(1, n_steps):
            for smpl in range(n_samples):
                if not valid[step, smpl]:
                    continue
                self.p_next_state[states[step, smpl], states[step-1, smpl], inputs[step, smpl]] += 1

                # print "from state=%d, to state=%d, input=%d" % (states[step-1, smpl], states[step, smpl], inputs[step, smpl])
                # print self.log_p_next_state[states[step, smpl], states[step-1, smpl], inputs[step, smpl]]

        # normalize
        sums = np.sum(self.p_next_state, axis=(0, 2))
        self.p_next_state /= sums[np.newaxis, :, np.newaxis]
        self.p_next_state[np.isnan(self.p_next_state)] = 0

    @staticmethod
    def discrete_normal(loc, scale, num_states):
        cdf = lambda x: scipy.stats.norm.cdf(x, loc=loc, scale=scale)

        low_lim = 0
        high_lim = num_states

        pos = np.arange(low_lim, high_lim+1)
        cdt = cdf(pos)
        pdt = np.diff(cdt)
        pdt[0] += cdf(low_lim)
        pdt[-1] += 1-cdf(high_lim)
        pos = pos[0:-1]

        #assert np.sum(pdt) == 1.0
        return pdt

    @staticmethod
    def nanargmax_12(x):
        mx = np.nanmax(x, axis=2)
        mmx = np.nanmax(mx, axis=1)
        am = np.nanargmax(mx, axis=1)

        amsel = x[np.arange(x.shape[0]), am, :]
        amx = np.nanargmax(amsel, axis=1)

        # print "mx:", mx
        # print "am:", am
        # print "sel:", x[range(x.shape[0]), am, :]
        # print "amx:", amx

        return mmx, am, amx

    def most_probable_states(self, inputs):
        # inputs[step]
        # msg[s_(step-1)]
        # max_track_s[step, s_step]
        # max_track_f[step, s_step]
        # max_state_s[step]

        n_steps = inputs.shape[0]
        assert np.all(0 <= inputs) and np.all(inputs < self.n_input_values)

        # track maximum states
        max_track_s = np.zeros((n_steps, self.n_system_states), dtype='int')
        max_track_f = np.zeros((n_steps, self.n_system_states), dtype='int')

        # leaf factor at beginning of chain
        msg = np.log(self.p_initial_state)

        # pass messages towards end node of chain
        # log_p_next_state[s_t, s_(t-1), f_t] = log p(s_t, f_t | s_(t-1))
        log_p_next_state = np.log(self.p_next_state)
        for step in range(n_steps):
            #print step
            # print
            # print self.log_p_next_state[:, :, inputs[step]]
            # print np.all(np.isnan(self.log_p_next_state[:, :, inputs[step]]))
            # print msg[np.newaxis, :]

            # m[s_t, s_(t-1), f_t]
            m1 = log_p_next_state
            m2 = msg
            m3 = np.log(self.discrete_normal(inputs[step], self.input_std, self.n_input_values))
            m = m1 + m2[np.newaxis, :, np.newaxis] + m3[np.newaxis, np.newaxis, :]
            msg, max_track_s[step, :], max_track_f[step, :] = self.nanargmax_12(m)
            #msg = np.nanmax(np.nanmax(m, axis=2), axis=1)
            #print step, ":", np.argmax(m, axis=1)

        # calculate maximum probability
        log_p_max = np.nanmax(msg)

        # backtrack maximum states
        max_state_s = np.zeros(n_steps, dtype='int')
        max_state_f = np.zeros(n_steps, dtype='int')
        max_state_s[n_steps-1] = np.nanargmax(msg)
        max_state_f[n_steps-1] = max_track_f[n_steps-1, max_state_s[n_steps-1]]
        for step in range(n_steps-2, -1, -1):
            max_state_s[step] = max_track_s[step+1, max_state_s[step+1]]
            max_state_f[step] = max_track_f[step, max_state_s[step]]

        return max_state_s, max_state_f, log_p_max

    def log_p(self, states, filtered_inputs, inputs):
        n_steps = inputs.shape[0]
        assert np.all(0 <= states) and np.all(states < self.n_system_states)
        assert np.all(0 <= filtered_inputs) and np.all(filtered_inputs < self.n_input_values)
        assert np.all(0 <= inputs) and np.all(inputs < self.n_input_values)

        lp = np.log(self.p_initial_state[states[0]])
        for step in range(1, n_steps):
            lp += np.log(self.p_next_state[states[step], states[step-1], filtered_inputs[step]])
            lp += np.log(self.discrete_normal(filtered_inputs[step], self.input_std, self.n_input_values)[inputs[step]])
        return lp

########################################################################################################################
########################################################################################################################
########################################################################################################################


class JointChain(object):

    def __init__(self, n_system_states, n_input_values):
        # log_p_next_state[s_t, s_(t-1), f_t] = p(s_t, f_t | s_(t-1))
        self.log_p_next_state = np.zeros((n_system_states, n_system_states, n_input_values))

        # log_p_initial_state[s_0] = p(s_0)
        self.log_p_initial_state = np.zeros((n_system_states,))

    @property
    def n_system_states(self):
        return self.log_p_next_state.shape[0]

    @property
    def n_input_values(self):
        return self.log_p_next_state.shape[2]

    def train_gaussian(self, states, inputs, valid):
        # states[step, sample]
        # inputs[step, sample]
        # valid[step, sample]

        n_steps = states.shape[0]
        n_samples = states.shape[1]
        assert np.all(0 <= states) and np.all(states < self.n_system_states)
        assert np.all(0 <= inputs) and np.all(inputs < self.n_input_values)
        assert states.dtype == 'int' and inputs.dtype == 'int'

        # initial state probabilities
        mu, sigma = normal_fit(states[0, :][np.newaxis, :])
        s = np.arange(self.n_system_states)
        self.log_p_initial_state[:] = normal_pdf(s[np.newaxis, :], mu, sigma)
        self.log_p_initial_state /= np.sum(self.log_p_initial_state)
        self.log_p_initial_state = np.log(self.log_p_initial_state)

        # state transition probabilities
        self.log_p_next_state[:] = 0
        for pstate in range(self.n_system_states):
            mask = np.roll((states == pstate) & valid, 1, axis=0) & valid
            if not np.any(mask):
                continue

            si = np.vstack((states[mask], inputs[mask]))
            mu, sigma = normal_fit(si)

            s, i = np.meshgrid(np.arange(self.n_system_states), np.arange(self.n_input_values))
            si = np.vstack((np.ravel(s.T), np.ravel(i.T)))
            sigma += 0.00001 * np.identity(2)
            pdf = normal_pdf(si, mu, sigma)
            self.log_p_next_state[:, pstate, :] = np.reshape(pdf, (self.n_system_states, self.n_input_values))
            self.log_p_next_state[:, pstate, :] /= np.sum(self.log_p_next_state[:, pstate, :])
        self.log_p_next_state = np.log(self.log_p_next_state)

    def train_gaussian_mixture(self, states, inputs, valid, n_components, covariance_type):
        # states[step, sample]
        # inputs[step, sample]
        # valid[step, sample]

        n_steps = states.shape[0]
        n_samples = states.shape[1]
        assert np.all(0 <= states) and np.all(states < self.n_system_states)
        assert np.all(0 <= inputs) and np.all(inputs < self.n_input_values)
        assert states.dtype == 'int' and inputs.dtype == 'int'

        # initial state probabilities
        mu, sigma = normal_fit(states[0, :][np.newaxis, :])
        s = np.arange(self.n_system_states)
        self.log_p_initial_state[:] = normal_pdf(s[np.newaxis, :], mu, sigma)
        self.log_p_initial_state /= np.sum(self.log_p_initial_state)
        self.log_p_initial_state = np.log(self.log_p_initial_state)

        # state transition probabilities
        self.log_p_next_state[:] = 0
        for pstate in range(self.n_system_states):
            mask = np.roll((states == pstate) & valid, 1, axis=0) & valid
            if not np.any(mask):
                continue

            si = np.vstack((states[mask], inputs[mask]))
            gmm = GMM(n_components, covariance_type)
            gmm.fit(si.T)

            s, i = np.meshgrid(np.arange(self.n_system_states), np.arange(self.n_input_values))
            si = np.vstack((np.ravel(s.T), np.ravel(i.T)))
            pdf = np.exp(gmm.score(si.T))
            self.log_p_next_state[:, pstate, :] = np.reshape(pdf, (self.n_system_states, self.n_input_values))
            self.log_p_next_state[:, pstate, :] /= np.sum(self.log_p_next_state[:, pstate, :])
        self.log_p_next_state = np.log(self.log_p_next_state)

    def train_kde(self, states, inputs, valid, kernel, bandwidth):
        # states[step, sample]
        # inputs[step, sample]
        # valid[step, sample]

        n_steps = states.shape[0]
        n_samples = states.shape[1]
        assert np.all(0 <= states) and np.all(states < self.n_system_states)
        assert np.all(0 <= inputs) and np.all(inputs < self.n_input_values)
        assert states.dtype == 'int' and inputs.dtype == 'int'

        # initial state probabilities
        mu, sigma = normal_fit(states[0, :][np.newaxis, :])
        s = np.arange(self.n_system_states)
        self.log_p_initial_state[:] = normal_pdf(s[np.newaxis, :], mu, sigma)
        self.log_p_initial_state /= np.sum(self.log_p_initial_state)
        self.log_p_initial_state = np.log(self.log_p_initial_state)

        # state transition probabilities
        self.log_p_next_state[:] = 0
        for pstate in range(self.n_system_states):
            mask = np.roll((states == pstate) & valid, 1, axis=0) & valid
            if not np.any(mask):
                continue
            print pstate

            si = np.vstack((states[mask], inputs[mask]))

            kde = KernelDensity(kernel=kernel, bandwidth=bandwidth, rtol=1e-4)
            # print "fit ", si.shape
            kde.fit(si.T)

            s, i = np.meshgrid(np.arange(self.n_system_states), np.arange(self.n_input_values))
            si = np.vstack((np.ravel(s.T), np.ravel(i.T)))
            # print "estimate ", si.shape
            pdf = np.exp(kde.score_samples(si.T))
            self.log_p_next_state[:, pstate, :] = np.reshape(pdf, (self.n_system_states, self.n_input_values))
            self.log_p_next_state[:, pstate, :] /= np.sum(self.log_p_next_state[:, pstate, :])
        self.log_p_next_state = np.log(self.log_p_next_state)

    def most_probable_states(self, inputs):
        # inputs[step]
        # msg[s_(step-1)]
        # max_track[step, s_step]
        # max_state[step]

        n_steps = inputs.shape[0]
        assert np.all(0 <= inputs) and np.all(inputs < self.n_input_values)

        # track maximum states
        max_track = np.zeros((n_steps, self.n_system_states), dtype='int')

        # leaf factor at beginning of chain
        msg = self.log_p_initial_state

        # pass messages towards end node of chain
        for step in range(n_steps):
            # m[s_t, s_(t-1)]
            m = self.log_p_next_state[:, :, inputs[step]] + msg[np.newaxis, :]
            max_track[step, :] = np.argmax(m, axis=1)
            msg = np.max(m, axis=1)

        # calculate maximum probability
        log_p_max = np.max(msg)

        # backtrack maximum states
        max_state = np.zeros(n_steps, dtype='int')
        max_state[n_steps-1] = np.argmax(msg)
        for step in range(n_steps-2, -1, -1):
            max_state[step] = max_track[step+1, max_state[step+1]]

        return max_state, log_p_max

    def log_p(self, states, inputs):
        n_steps = inputs.shape[0]
        assert np.all(0 <= states) and np.all(states < self.n_system_states)
        assert np.all(0 <= inputs) and np.all(inputs < self.n_input_values)

        lp = self.log_p_initial_state[states[0]]
        for step in range(1, n_steps):
            lp += self.log_p_next_state[states[step], states[step-1], inputs[step]]
        return lp

