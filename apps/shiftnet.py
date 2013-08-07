# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import common.gpu

import climin
import numpy as np
import theano
import theano.tensor as T
import breze.util
import matplotlib.pyplot as plt

import common.util
import nn.gpushift
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
cfg, plot_dir = common.util.standard_cfg()

# parameters
ps = breze.util.ParameterSet(**FourierShiftNet.parameter_shapes(cfg.x_len, cfg.s_len))

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

f_trn_loss = lambda p: f_loss(p, trn_inputs, trn_shifts, trn_targets)
f_trn_dloss = lambda p: f_dloss(p, trn_inputs, trn_shifts, trn_targets)

# generate data
print "Generating data..."
if cfg.generate_on_gpu:
    trn_inputs, trn_shifts, trn_targets = nn.gpushift.generate_data(cfg.x_len, cfg.s_len, cfg.n_samples)
    val_inputs, val_shifts, val_targets = nn.gpushift.generate_data(cfg.x_len, cfg.s_len, cfg.n_samples)
    tst_inputs, tst_shifts, tst_targets = nn.gpushift.generate_data(cfg.x_len, cfg.s_len, cfg.n_samples)
else:
    trn_inputs, trn_shifts, trn_targets = nn.shift.generate_data(cfg.x_len, cfg.s_len, cfg.n_samples)
    val_inputs, val_shifts, val_targets = nn.shift.generate_data(cfg.x_len, cfg.s_len, cfg.n_samples)
    tst_inputs, tst_shifts, tst_targets = nn.shift.generate_data(cfg.x_len, cfg.s_len, cfg.n_samples)

    trn_inputs = post(trn_inputs)
    trn_shifts = post(trn_shifts)
    trn_targets = post(trn_targets)
    val_inputs = post(val_inputs)
    val_shifts = post(val_shifts)
    val_targets = post(val_targets)
    tst_inputs = post(tst_inputs)
    tst_shifts = post(tst_shifts)
    tst_targets = post(tst_targets)
print "Done."                                 

# optimizer
if cfg.optimizer == 'lbfgs':
    opt = climin.Lbfgs(ps.data, f_trn_loss, f_trn_dloss)
elif cfg.optimizer == 'rprop':
    opt = climin.Rprop(ps.data, f_trn_loss, f_trn_dloss)
elif cfg.optimizer == 'rmsprop':
    opt = climin.RmsProp(ps.data, f_trn_dloss, steprate=cfg.steprate, momentum=cfg.momentum)
elif cfg.optimizer == 'gradientdescent':
    opt = climin.GradientDescent(ps.data, f_trn_dloss, steprate=cfg.steprate)
else:
    assert False, "unknown optimizer"

# initialize parameters
ps.data[:] = cfg.init * (np.random.random(ps.data.shape) - 0.5)
#n_activate = int(len(ps.data) / 10)
#for n in range(n_activate):
#    i = np.random.randint(len(ps.data))
#    ps.data[i] = 1
print "initial loss: ", f_loss(ps.data, trn_inputs, trn_shifts, trn_targets)

# optimize
his = common.util.ParameterHistory(max_missed_val_improvements=1000,
                                   desired_loss=0.0001)
for iter, sts in enumerate(opt):
    if iter % 10 == 0:
        trn_loss = gather(f_loss(ps.data, trn_inputs, trn_shifts, trn_targets))
        val_loss = gather(f_loss(ps.data, val_inputs, val_shifts, val_targets))
        tst_loss = gather(f_loss(ps.data, tst_inputs, tst_shifts, tst_targets))
        
        his.add(iter, ps.data, trn_loss, val_loss, tst_loss)
        if his.should_terminate:
            break
          
# save results            
ps.data[:] = his.best_pars
his.plot()
plt.savefig(plot_dir + "/loss.pdf")

# check with simple patterns
sim_inputs, sim_shifts, sim_targets = generate_data(cfg.x_len, cfg.s_len, 3, binary=True)
sim_results = gather(f_output(ps.data, post(sim_inputs), post(sim_shifts)))
print "input:   "
print sim_inputs.T
print "shift:   "
print sim_shifts.T
print "targets: "
print sim_targets.T
print "results: "
print sim_results.T

# profiler output
if profile:
    profmode.print_summary()

