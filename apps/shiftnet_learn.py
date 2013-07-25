# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import climin
import numpy as np
import theano
import theano.tensor as T
import breze.util

import common.util
from nn.shift import *
from common.complex import *
from common.util import floatx
from common.gpu import gather, post, function
from math import floor

theano.config.compute_test_value = 'ignore'
np.set_printoptions(precision=3, suppress=True)

# hyperparameters
x_len = 20
s_len = x_len
n_samples = 10000

# parameters
ps = breze.util.ParameterSet(**FourierShiftNet.parameter_shapes(x_len, s_len))

# inputs
x = T.matrix('x')
#x.tag.test_value = np.random.random((x_len, n_samples))
s = T.matrix('s')
#s.tag.test_value = np.random.random((s_len, n_samples))
t = T.matrix('t')
#t.tag.test_value = np.random.random((x_len, n_samples))

# functions
fsn = FourierShiftNet(**ps.vars)
f_output = function(inputs=[ps.flat,x,s], outputs=fsn.output(x,s))

loss = T.mean((fsn.output(x,s) - t)**2)
f_loss = function(inputs=[ps.flat,x,s,t], outputs=loss)
f_dloss = function(inputs=[ps.flat,x,s,t], outputs=T.grad(loss, ps.flat))

# generate data
trn_inputs, trn_shifts, trn_targets = generate_data(x_len, s_len, n_samples)
val_inputs, val_shifts, val_targets = generate_data(x_len, s_len, n_samples)
tst_inputs, tst_shifts, tst_targets = generate_data(x_len, s_len, n_samples)

# Training 
ps.data[:] = 0.01 * (np.random.random(ps.data.shape) - 0.5)
his = common.util.ParameterHistory(max_missed_val_improvements=100,
                                   desired_loss=0.0001)

f_trn_loss = lambda p: f_loss(p, trn_inputs, trn_shifts, trn_targets)
f_trn_dloss = lambda p: f_dloss(p, trn_inputs, trn_shifts, trn_targets)

# optimizer
#opt = climin.Lbfgs(ps.data, f_trn_loss, f_trn_dloss)
opt = climin.Rprop(ps.data, f_trn_loss, f_trn_dloss)
#opt = climin.GradientDescent(ps.data, f_trn_dloss, steprate=0.01)

print "initial loss: ", f_loss(ps.data, trn_inputs, trn_shifts, trn_targets)

for iter, sts in enumerate(opt):
    if iter % 10 == 0:
        trn_loss = f_loss(ps.data, trn_inputs, trn_shifts, trn_targets)
        val_loss = f_loss(ps.data, val_inputs, val_shifts, val_targets)
        tst_loss = f_loss(ps.data, tst_inputs, tst_shifts, tst_targets)
        
        his.add(iter, ps.data, trn_loss, val_loss, tst_loss)
        if his.should_terminate:
            break
                      
ps.data[:] = his.best_pars
his.plot()

# check with simple patterns
sim_inputs, sim_shifts, sim_targets = generate_data(x_len, s_len, 3, binary=True)
sim_results = f_output(ps.data, sim_inputs, sim_shifts)

print "input:   "
print sim_inputs.T
print "shift:   "
print sim_shifts.T
print "targets: "
print sim_targets.T
print "results: "
print sim_results.T



