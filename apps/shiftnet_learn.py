# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import climin
import numpy as np
import theano
import theano.tensor as T
import breze.util
import matplotlib.pyplot as plt

import common.util
from nn.shift import *
from common.complex import *
from common.util import floatx
from common.gpu import gather, post, function
from math import floor

if theano.config.device != 'gpu':
    theano.config.compute_test_value = 'ignore'
np.set_printoptions(precision=3, suppress=True)

profile = False

if profile:
    from theano import ProfileMode
    profmode = theano.ProfileMode(optimizer='fast_run', 
                                  linker=theano.gof.OpWiseCLinker())

# hyperparameters
x_len = 80
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
if profile:
    f_loss = function(inputs=[ps.flat,x,s,t], outputs=loss, mode=profmode, name='f_loss')
    f_dloss = function(inputs=[ps.flat,x,s,t], outputs=T.grad(loss, ps.flat), mode=profmode, name='f_dloss')
else:
    f_loss = function(inputs=[ps.flat,x,s,t], outputs=loss)
    f_dloss = function(inputs=[ps.flat,x,s,t], outputs=T.grad(loss, ps.flat))

# generate data

trn_inputs, trn_shifts, trn_targets = generate_data(x_len, s_len, n_samples)
val_inputs, val_shifts, val_targets = generate_data(x_len, s_len, n_samples)
tst_inputs, tst_shifts, tst_targets = generate_data(x_len, s_len, n_samples)

# transfer to GPU
trn_inputs = post(trn_inputs)
trn_shifts = post(trn_shifts)
trn_targets = post(trn_targets)
val_inputs = post(val_inputs)
val_shifts = post(val_shifts)
val_targets = post(val_targets)
tst_inputs = post(tst_inputs)
tst_shifts = post(tst_shifts)
tst_targets = post(tst_targets)

# Training 
ps.data[:] = 0.01 * (np.random.random(ps.data.shape) - 0.5)
#ps.data[:] = 0.001 * (np.random.random(ps.data.shape) - 0.5)
#n_activate = int(len(ps.data) / 10)
#for n in range(n_activate):
#    i = np.random.randint(len(ps.data))
#    ps.data[i] = 1
    
his = common.util.ParameterHistory(max_missed_val_improvements=1000,
                                   desired_loss=0.0001)
                                   

f_trn_loss = lambda p: f_loss(p, trn_inputs, trn_shifts, trn_targets)
f_trn_dloss = lambda p: f_dloss(p, trn_inputs, trn_shifts, trn_targets)

# optimizer
#opt = climin.Lbfgs(ps.data, f_trn_loss, f_trn_dloss)
#opt = climin.Rprop(ps.data, f_trn_loss, f_trn_dloss)
opt = climin.RmsProp(ps.data, f_trn_dloss, 0.001)
#opt = climin.GradientDescent(ps.data, f_trn_dloss, steprate=0.01)

print "initial loss: ", f_loss(ps.data, trn_inputs, trn_shifts, trn_targets)

for iter, sts in enumerate(opt):
    if iter % 10 == 0:
        trn_loss = gather(f_loss(ps.data, trn_inputs, trn_shifts, trn_targets))
        val_loss = gather(f_loss(ps.data, val_inputs, val_shifts, val_targets))
        tst_loss = gather(f_loss(ps.data, tst_inputs, tst_shifts, tst_targets))
        
        his.add(iter, ps.data, trn_loss, val_loss, tst_loss)
        if his.should_terminate:
            break
                      
ps.data[:] = his.best_pars
his.plot()
plt.savefig("shiftnet_learn.pdf")

# check with simple patterns
sim_inputs, sim_shifts, sim_targets = generate_data(x_len, s_len, 3, binary=True)
sim_results = gather(f_output(ps.data, post(sim_inputs), post(sim_shifts)))

print "input:   "
print sim_inputs.T
print "shift:   "
print sim_shifts.T
print "targets: "
print sim_targets.T
print "results: "
print sim_results.T


if profile:
    profmode.print_summary()

