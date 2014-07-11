from __future__ import division

import numpy as np
from scipy.misc import logsumexp
import scipy.stats
import matplotlib.pyplot as plt
import GPy as gpy

from GPy.util import Tango
from sklearn.mixture import GMM, VBGMM
from sklearn.neighbors.kde import KernelDensity
from ml.common.progress import status, done


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


def normal_pdf(x, mu, sigma):
    assert x.ndim == 2
    assert mu.ndim == 1 and sigma.ndim == 2
    k = x.shape[0]
    assert k == mu.shape[0]
    assert k == sigma.shape[0] == sigma.shape[1]

    nf = (2.0*np.pi)**(-k/2.0) * np.linalg.det(sigma)**(-0.5)
    prec = np.linalg.inv(sigma)
    e = -0.5 * np.sum((x - mu[:, np.newaxis]) * np.dot(prec, (x - mu[:, np.newaxis])), axis=0)
    pdf = nf * np.exp(e)

    return pdf


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
    assert valid.ndim == 2

    n_inps = len(inps)
    n_steps = valid.shape[0]
    n_samples = valid.shape[1]

    for inp in inps:
        assert inp.shape[-1] == n_samples

    outs = []

    for smpl in range(n_samples):
        n_valid = np.where(valid[:, smpl])[0][-1]

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

    if len(outs) == 1:
        return outs[0]
    else:
        return tuple(outs)


class VarGP(object):

    def __init__(self, inp, tar,
                 use_data_variance=True,
                 kernel=gpy.kern.rbf, normalize_x=True, normalize_y=True,
                 optimize=True, optimize_restarts=1,
                 gp_parameters={},
                 cutoff_stds=None, std_adjust=1.0, std_power=1.0, min_std=0.01):

        # inp[feature, smpl]
        # tar[smpl]

        assert inp.ndim == 2
        assert tar.ndim == 1

        self.mngful = None
        self.cutoff_stds = cutoff_stds
        self.std_adjust = std_adjust
        self.std_power = std_power
        self.use_data_variance = use_data_variance
        self.min_std = min_std
        self.kernel = kernel
        self.normalize_x = normalize_x
        self.normalize_y = normalize_y
        self.gp_parameters = gp_parameters
        self.n_features = None

        self.gp = None
        self.gp_var = None

        self.train(inp, tar, optimize, optimize_restarts)

    def copy_gp_parameters(self, src, dest):
        for par in src._get_param_names():
            dest[par] = src[par]

    def train(self, inp, tar, optimize=False, optimize_restarts=1):
        # merge duplicates
        inp_mrgd, sta_mrgd, sta_var = self._merge_duplicates(inp, tar)

        # handle singleton data (workaround for non-positive definite matrices in GPy)
        if inp_mrgd.shape[1] == 1:
            inp_mrgd = np.resize(inp_mrgd, (inp_mrgd.shape[0], 2))
            sta_mrgd = np.resize(sta_mrgd, (2,))
            inp_mrgd[:, 1] = inp_mrgd[:, 0] + 1
            sta_mrgd[1] = sta_mrgd[0]
            singleton_fix = True
        else:
            singleton_fix = False

        n_samples = inp_mrgd.shape[1]
        self.n_features = inp_mrgd.shape[0]

        old_gp = self.gp
        old_gp_var = self.gp_var

        # fit GP on data
        self.gp = gpy.models.GPRegression(inp_mrgd.T, sta_mrgd[:, np.newaxis], self.kernel(),
                                          normalize_X=self.normalize_x, normalize_Y=self.normalize_y)
        self.gp.unconstrain('')
        for k, v in self.gp_parameters.iteritems():
            if v is not None:
                self.gp.constrain_fixed(k, v)
        self.gp.ensure_default_constraints()
        if old_gp is not None:
            self.copy_gp_parameters(old_gp, self.gp)
        if optimize:
            if optimize_restarts == 1:
                self.gp.optimize()
            else:
                self.gp.optimize_restarts(num_restarts=optimize_restarts)

        # estimate variance from predictions of first GP
        sta_mu, _, _, _ = self.gp.predict(inp_mrgd.T, full_cov=False)
        sta_mrgd_var = np.zeros(n_samples)
        for n in range(n_samples):
            smpl_inp = np.all(np.isclose(inp, inp_mrgd[:, n:n+1]), axis=0)
            sta_mrgd_var[n] = np.mean((tar[smpl_inp] - sta_mu[n])**2)
        if singleton_fix:
            sta_mrgd_var = np.zeros(shape=sta_mrgd.shape)

        # fit GP on variance
        self.gp_var = gpy.models.GPRegression(inp_mrgd.T, sta_mrgd_var[:, np.newaxis], self.kernel(),
                                              normalize_Y=True)
        self.gp_var.unconstrain('')
        self.gp_var.ensure_default_constraints()
        if old_gp_var is not None:
            self.copy_gp_parameters(old_gp_var, self.gp_var)
        if optimize:
            self.gp_var.optimize()

    def __str__(self):
        return str(self.gp)

    def _merge_duplicates(self, inp, tar):
        # inp[feature, smpl]
        # tar[smpl]

        assert tar.ndim == 1
        n_features = inp.shape[0]
        n_samples = inp.shape[1]

        xn = {}
        for smpl in range(n_samples):
            k = tuple(inp[:, smpl])
            if k in xn:
                xn[k].append(tar[smpl])
            else:
                xn[k] = [tar[smpl]]

        merged_inp = np.zeros((n_features, len(xn.keys())), dtype=inp.dtype)
        merged_tar = np.zeros(len(xn.keys()))
        merged_tar_var = np.zeros(len(xn.keys()))
        for smpl, (k, v) in enumerate(xn.iteritems()):
            merged_inp[:, smpl] = np.asarray(k)
            merged_tar[smpl] = np.mean(v)
            merged_tar_var[smpl] = np.var(v, ddof=1)

        return merged_inp, merged_tar, merged_tar_var

    def limit_meaningful_predictions(self, mngful_dist, inp, n_inp_values):
        # inp[feature, smpl]

        assert inp.ndim == 2
        assert inp.shape[0] == self.n_features
        assert self.n_features == 1, "limit_meaningful_predictions() requires one-dimensional input space"

        # determine meaningful predictions
        self.mngful = np.zeros(n_inp_values, dtype='bool')
        for i in range(mngful_dist):
            self.mngful[np.minimum(inp[0, :] + i, n_inp_values - 1)] = True
            self.mngful[np.maximum(inp[0, :] - i, 0)] = True

    def predict(self, inp):
        assert inp.ndim == 2
        assert inp.shape[0] == self.n_features

        tar, var_gp, _, _ = self.gp.predict(inp.T, full_cov=False)
        tar = tar[:, 0]
        var_gp = var_gp[:, 0]

        var_data, _, _, _ = self.gp_var.predict(inp.T, full_cov=False)
        var_data = var_data[:, 0]
        var_data[var_data < 0] = 0

        if self.use_data_variance:
            std = np.sqrt(var_gp) + np.sqrt(var_data)
        else:
            std = np.sqrt(var_gp)
        std = self.std_adjust * np.power(std, self.std_power)

        return tar, std, var_gp, var_data

    def pdf_for_all_inputs(self, n_inp_values, n_tar_values):
        assert self.n_features == 1, "pdf_for_all_inputs() requires one-dimensional input space"
        inp = np.arange(n_inp_values)[np.newaxis, :]
        return self.pdf(inp, n_tar_values)

    def pdf(self, inp, n_tar_values):
        # inp[feature, smpl]
        # pdf[tar, smpl]
        # tar[smpl]
        # std[smpl]

        assert inp.ndim == 2
        assert inp.shape[0] == self.n_features

        n_samples = inp.shape[1]

        # calculate predictions and associated variances
        tar, std, _, _ = self.predict(inp)
        std[std < self.min_std] = self.min_std

        # discretize output distribution
        pdf = np.zeros((n_tar_values, n_samples))
        tar_values = np.arange(n_tar_values)
        for smpl in range(n_samples):
            if self.mngful is not None:
                inp_val = inp[0, smpl]
                if not self.mngful[inp_val]:
                    continue

            ipdf = normal_pdf(tar_values[np.newaxis, :], np.atleast_1d(tar[smpl]), np.atleast_2d(std[smpl])**2)
            if self.cutoff_stds is not None:
                ipdf[np.abs(tar_values - tar[smpl]) > self.cutoff_stds * std[smpl]] = 0
            nfac = np.sum(ipdf)
            if nfac == 0:
                nfac = 1
            ipdf /= nfac
            pdf[:, smpl] = ipdf

        return pdf

    def plot(self, trn_inp, trn_tar, n_inp_values, n_tar_values, rng=None, hmarker=None):
        assert self.n_features == 1, "plot() requires one-dimensional input space"

        if rng is None:
            rng = [0, n_inp_values]

        inp = np.arange(n_inp_values)[np.newaxis, :]
        tar, std, var_gp, var_data = self.predict(inp)
        pdf = self.pdf_for_all_inputs(n_inp_values, n_tar_values)

        # plot base GP
        plt.subplot(2, 2, 1)
        plt.hold(True)
        self.gp.plot(plot_limits=rng, which_data_rows=[], ax=plt.gca())
        plt.ylim(0, n_tar_values)
        plt.plot((n_tar_values - 5) * self.mngful, 'g')
        plt.plot(trn_inp[0, :], trn_tar, 'r.')
        if hmarker is not None:
            plt.axhline(hmarker, color='r')

        # plot GP with data variance
        plt.subplot(2, 2, 2)
        plt.hold(True)
        plt.ylim(0, n_tar_values)
        conf_gp = 2 * np.sqrt(var_gp)
        conf_total = 2 * std
        # cutoff = cutoff_stds * pred_std
        # plt.fill_between(pred_inp, pred_tar + cutoff, pred_tar - cutoff,
        #                  edgecolor=Tango.colorsHex['darkRed'], facecolor=Tango.colorsHex['lightRed'], alpha=0.4)
        plt.fill_between(inp[0, :], tar + conf_total, tar - conf_total,
                         edgecolor=Tango.colorsHex['darkBlue'], facecolor=Tango.colorsHex['lightBlue'], alpha=0.4)
        plt.fill_between(inp[0, :], tar + conf_gp, tar - conf_gp,
                         edgecolor=Tango.colorsHex['darkPurple'], facecolor=Tango.colorsHex['lightPurple'], alpha=0.4)
        plt.plot(inp[0, :], tar, 'k')
        #plt.errorbar(pred_inp, pred_tar, 2*np.sqrt(sta_mrgd_var), fmt=None)
        if hmarker is not None:
            plt.axhline(hmarker, color='r')

        # plot output distribution
        plt.subplot(2, 2, 3)
        plt.imshow(pdf, origin='lower', aspect=1, interpolation='none')
        plt.colorbar()


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
            p_next_input[:, pinp] = scipy.stats.norm.pdf_for_all_inputs(all_inputs, loc=pinp, scale=sigma)
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

        # GPs that model p(s_t | s_(t-1), f_t)
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

    def infer_states(self, observations):
        # observations[feature, step]
        # states[step]
        # states_std[step]

        states, states_std, _, _ = self.gp_states_given_observations.predict(observations)
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

