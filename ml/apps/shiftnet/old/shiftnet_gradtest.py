# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import breze.util

from ml.nn.shift import *
from ml.common.complex import *

if theano.config.device != 'gpu':
    theano.config.compute_test_value = 'ignore'
np.set_printoptions(precision=3, suppress=True)

profile = True

if profile:
    profmode = theano.ProfileMode(optimizer='fast_run',
                                  linker=theano.gof.OpWiseCLinker())

# hyperparameters
x_len = 20
s_len = x_len
n_samples = 1000

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

f_trn_loss = lambda p: f_loss(p, trn_inputs, trn_shifts, trn_targets)
f_trn_dloss = lambda p: f_dloss(p, trn_inputs, trn_shifts, trn_targets)


#print "calculating loss"
#x = f_loss(ps.data, trn_inputs, trn_shifts, trn_targets)

print "calculating gradient"
dx = f_dloss(ps.data, trn_inputs, trn_shifts, trn_targets)



if profile:
    profmode.print_summary()

