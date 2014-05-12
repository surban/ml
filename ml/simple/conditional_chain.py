from __future__ import division

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import GPy as gpy

from sklearn.mixture import GMM, VBGMM
from sklearn.neighbors.kde import KernelDensity


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
    i, s = merge_duplicates(i, s)

    idx = np.argsort(i)
    i = i[idx]
    s = s[idx]

    plt.plot(i, s, style)


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

    xout = np.zeros(len(xn.keys()))
    yout = np.zeros(xout.shape)
    for i, (k, v) in enumerate(xn.iteritems()):
        xout[i] = k
        yout[i] = np.mean(v)

    return xout, yout


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

    def train_gp(self, states, inputs, valid, limit_sigma,
                 gp_optimize, gp_variance=None, gp_lengthscale=None, gp_noise=None):
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
                # print "skipping state %d" % pstate
                continue

            i = inputs[mask]
            s = states[mask]
            i, s = merge_duplicates(i, s)
            # print "pstate=%d #points=%d" % (pstate, i.size)
            # print i
            # print s

            kernel = gpy.kern.rbf(input_dim=1)
            gp = gpy.models.GPRegression(i[:, np.newaxis], s[:, np.newaxis], kernel)
            gp.unconstrain('')
            if gp_variance is not None:
                gp.constrain_fixed('rbf_variance', gp_variance)
            if gp_lengthscale is not None:
                gp.constrain_fixed('rbf_lengthscale', gp_lengthscale)
            if gp_noise is not None:
                gp.constrain_fixed('noise_variance', gp_noise)
            gp.ensure_default_constraints()
            if gp_optimize:
                gp.optimize()
            # print gp

            cs = np.arange(self.n_system_states)
            ci = np.arange(self.n_input_values)
            s_mu, s_var, _, _ = gp.predict(ci[:, np.newaxis], full_cov=False)
            s_mu = s_mu[:, 0]
            s_var = s_var[:, 0]
            s_sigma = np.sqrt(s_var)
            # s_sigma += noise_sigma
            for inp in range(self.n_input_values):
                if s_sigma[inp] < limit_sigma:
                    self.log_p_next_state[:, pstate, inp] = normal_pdf(cs[np.newaxis, :],
                                                                       np.atleast_1d(s_mu[inp]),
                                                                       np.atleast_2d(s_sigma[inp]))
                    self.log_p_next_state[:, pstate, inp] /= np.sum(self.log_p_next_state[:, pstate, inp])
                else:
                    self.log_p_next_state[:, pstate, inp] = 0

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

