from __future__ import division

import numpy as np


class ConditionalChain(object):

    def __init__(self, n_system_states, n_input_values):
        # p_next_state[s_t, s_(t-1), f_t] = p(s_t | s_(t-1), f_t)
        self.p_next_state = np.zeros((n_system_states, n_system_states, n_input_values))

        # p_initial_state[s_0] = p(s_0)
        self.p_initial_state = np.zeros((n_system_states,))

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
                # print self.p_next_state[states[step, smpl], states[step-1, smpl], inputs[step, smpl]]

        # identity prior
        #self.p_next_state += np.identity(self.n_system_states)[:, :, np.newaxis]

        # non-informative distribution for unobserved transitions
        # s = np.sum(self.p_next_state, axis=0)
        # a = np.zeros(s.shape)
        # a[s == 0] = 1
        # self.p_next_state += a

        # normalize
        # self.p_next_state += 1.0 / float(self.n_system_states)
        sums = np.sum(self.p_next_state, axis=0)
        # sums += 1
        self.p_next_state /= sums[np.newaxis, :, :]
        # self.p_next_state[np.isnan(self.p_next_state)] = 0

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
        msg = np.log(self.p_initial_state)

        # pass messages towards end node of chain
        for step in range(n_steps):
            # print
            # print self.p_next_state[:, :, inputs[step]]
            # print np.all(np.isnan(self.p_next_state[:, :, inputs[step]]))
            # print msg[np.newaxis, :]

            # m[s_t, s_(t-1)]
            m = np.log(self.p_next_state[:, :, inputs[step]]) + msg[np.newaxis, :]
            max_track[step, :] = np.nanargmax(m, axis=1)
            #print step, ":", np.argmax(m, axis=1)
            msg = np.nanmax(m, axis=1)

        # calculate maximum probability
        log_p_max = np.nanmax(msg)

        # backtrack maximum states
        max_state = np.zeros(n_steps, dtype='int')
        max_state[n_steps-1] = np.nanargmax(msg)
        # max_state[n_steps-1] = np.argmax(max_track[n_steps-1, :])
        for step in range(n_steps-2, -1, -1):
            max_state[step] = max_track[step+1, max_state[step+1]]

        return max_state, log_p_max

    def log_p(self, states, inputs):
        n_steps = inputs.shape[0]
        assert np.all(0 <= states) and np.all(states < self.n_system_states)
        assert np.all(0 <= inputs) and np.all(inputs < self.n_input_values)

        lp = np.log(self.p_initial_state[states[0]])
        for step in range(1, n_steps):
            lp += np.log(self.p_next_state[states[step], states[step-1], inputs[step]])
        return lp

    def greedy_states(self, inputs):
        n_steps = inputs.shape[0]
        assert np.all(0 <= inputs) and np.all(inputs < self.n_input_values)

        max_state = np.zeros(n_steps, dtype='int')
        cur_state = np.argmin(self.p_initial_state)
        for step in range(0, n_steps):
            # print "step=%d, state=%d, input=%d, p(s_t+1 | state, input)=" % (step, cur_state, inputs[step])
            # print self.p_next_state[:, cur_state, inputs[step]]

            cur_state = np.argmax(self.p_next_state[:, cur_state, inputs[step]])
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
            # print self.p_next_state[:, states[step-1], inputs[step]]

            max_state[step] = np.argmax(self.p_next_state[:, states[step-1], inputs[step]])

        return max_state
