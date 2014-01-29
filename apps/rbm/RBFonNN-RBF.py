# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import climin
import numpy as np
import gnumpy as gp
import theano
import theano.tensor as T
import breze.util
import time

import common.util
from nn.nn import OneHiddenLayerNN, StackedRBF
from common.util import floatx
from common.gpu import function, gather, post
from math import floor

# <codecell>

# load data
np.random.seed(100)
#RX, RZ, VX, VZ, TX, TZ = common.util.load_theano_data('../datasets/boston_split.mat')   
RX, RZ, VX, VZ, TX, TZ = common.util.load_theano_data('../datasets/abalone_split.mat') 

# <headingcell level=2>

# Proposal: RBF on sigmoid layer

# <codecell>

# check kernel
x=post(np.array([[11, 21, 31], [12, 22, 32]]))
y=post(np.array([[101, 201],   [102, 202]]))
l=post(np.array([[100]]))

tx = T.matrix('x')
ty = T.matrix('y')
tl = T.matrix('l')
f_kernel_matrix = function([tx, ty, tl], StackedRBF.kernel_matrix(tx, ty, tl))

K = f_kernel_matrix(x, y, l)
print gather(K)

# <codecell>

# hyperparameters
n_targets = RZ.get_value().shape[0]
n_features = RX.get_value().shape[0]
n_samples = RX.get_value().shape[1]
n_hidden = 50
#n_pivots = int(n_samples / 2)
n_pivots = 200

# <codecell>

# Theano expressions

# parameters
ps = breze.util.ParameterSet(P=(n_hidden, n_pivots),
                             W=(n_targets, n_pivots),
                             V=(n_hidden, n_features),
                             l=(1,1))

# expressions
srbf = StackedRBF(ps.P, ps.W, ps.V, ps.l)
RL = srbf.regression_objective(RX, RZ)

# functions
f_RL = function(inputs=[ps.flat], outputs=RL)
f_VL = function(inputs=[ps.flat], outputs=srbf.regression_objective(VX, VZ))
f_TL = function(inputs=[ps.flat], outputs=srbf.regression_objective(TX, TZ))
f_dRL = function(inputs=[ps.flat], outputs=T.grad(RL, ps.flat)) 

# <codecell>

# initialize parameters
ps.data[:] = np.random.random(ps.data.shape) - 0.5
ps['l'] = 2;

# test Theano expressions
print "Training loss:   %f" % gp.as_numpy_array(f_RL(ps.data))
print "Validation loss: %f" % gp.as_numpy_array(f_VL(ps.data))
print "Test loss:       %f" % gp.as_numpy_array(f_TL(ps.data))
gr = f_dRL(ps.data)
print "Gradient shape:    ", gr.shape

# <codecell>

# initialize 
ps.data[:] = np.random.random(ps.data.shape) - 0.5
ps['l'] = 100;
his = common.util.ParameterHistory(max_missed_val_improvements=100)

# optimize 
#opt = climin.Lbfgs(ps.data, f_RL, f_dRL)
opt = climin.Rprop(ps.data, f_RL, f_dRL)
#opt = climin.GradientDescent(ps.data, f_dRL, steprate=0.0001)


start = time.clock()
for iter, sts in enumerate(opt):
    if iter % 10 == 0:
        trn_loss = gp.as_numpy_array(f_RL(ps.data))
        val_loss = gp.as_numpy_array(f_VL(ps.data))
        tst_loss = gp.as_numpy_array(f_TL(ps.data))
        
        his.add(iter, ps.data, trn_loss, val_loss, tst_loss)
        if his.should_terminate:
            break
            
        #print "l = ", ps['l']
end = time.clock()
            
ps.data[:] = his.best_pars
his.plot()

# <codecell>

print "l: ", ps['l']

print "elapsed time: %f" % (end - start)


