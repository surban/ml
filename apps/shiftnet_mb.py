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

np.set_printoptions(precision=3, suppress=True)

# hyperparameters
cfg, plot_dir = common.util.standard_cfg()

# parameters
ps = breze.util.ParameterSet(**FourierShiftNet.parameter_shapes(cfg.x_len, cfg.s_len))

# inputs
x = T.matrix('x')
s = T.matrix('s')
t = T.matrix('t')

# functions
fsn = FourierShiftNet(**ps.vars)
f_output = function(inputs=[ps.flat,x,s], outputs=fsn.output(x,s))

loss = T.mean((fsn.output(x,s) - t)**2)
f_loss = function(inputs=[ps.flat,x,s,t], outputs=loss)
f_dloss = function(inputs=[ps.flat,x,s,t], outputs=T.grad(loss, ps.flat))

print "Generating validation data..."
val_inputs, val_shifts, val_targets = nn.gpushift.generate_data(cfg.x_len, cfg.s_len, cfg.n_val_samples)
tst_inputs, tst_shifts, tst_targets = nn.gpushift.generate_data(cfg.x_len, cfg.s_len, cfg.n_val_samples)
print "Done."
                                    
# optimizer
def f_trn_dloss(p):
    trn_inputs, trn_shifts, trn_targets = \
        nn.gpushift.generate_data(cfg.x_len, cfg.s_len, cfg.n_batch)
    return f_dloss(p, trn_inputs, trn_shifts, trn_targets)

if cfg.optimizer == 'rmsprop':
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
print "initial validation loss: ", f_loss(ps.data, val_inputs, val_shifts, val_targets)

# optimize
his = common.util.ParameterHistory(max_missed_val_improvements=1000,
                                   desired_loss=0.0001)
for iter, sts in enumerate(opt):
    if iter % 10 == 0:
        val_loss = gather(f_loss(ps.data, val_inputs, val_shifts, val_targets))
        tst_loss = gather(f_loss(ps.data, tst_inputs, tst_shifts, tst_targets))
        
        his.add(iter, ps.data, 1, val_loss, tst_loss)
        if his.should_terminate:
            break
                      
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



