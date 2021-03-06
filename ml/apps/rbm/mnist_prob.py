# -*- coding: utf-8 -*-

import numpy as np
import scipy.io

import gnumpy as gp
from ml.apps.rbm import mnist_rbm_config as cfg
import ml.rbm.util as rbmutil
from ml.rbm.rbm import RestrictedBoltzmannMachine


# numeric overflow handling
#np.seterr(all='raise')
#gp.acceptable_number_types = 'no nans or infs'

# parameters
epoch = cfg.epochs - 1
#epoch = 9
use_ruslan = False

# load dataset
X, TX = rbmutil.load_mnist(False)

# load ruslan's training set
mdata = scipy.io.loadmat("mnist.mat")
X = gp.as_garray(mdata['fbatchdata'])

# enter output directory
rbmutil.enter_rbm_plot_directory("mnist", cfg.n_hid, cfg.use_pcd, cfg.n_gibbs_steps,
                                 "prob.txt", clean=False)

# Build RBM
rbm = RestrictedBoltzmannMachine(0, cfg.n_vis, cfg.n_hid, 0) 

# load Ruslan's RBM
if use_ruslan:
    print "Loading Ruslan's ml.rbm..."
    mdata = scipy.io.loadmat("matlab_epoch%d.mat" % (epoch + 1))
    ml.rbm.bias_vis = gp.as_garray(mdata['visbiases'][0,:])
    ml.rbm.bias_hid = gp.as_garray(mdata['hidbiases'][0,:])
    ml.rbm.weights = gp.as_garray(mdata['vishid'])
else:
    rbmutil.load_parameters(rbm, "weights-%02i.npz" % epoch)

# load pratition function
if use_ruslan:
    filename = "matlab-lpf-%02d.npz" % (epoch+1)
else:
    filename = "lpf-%02d.npz" % epoch
print "Loading partition function %s" % filename
lpf = np.load(filename)
rbm.log_pf = lpf['lpf']

# calculate log probability of training set
tr_lp = gp.mean(rbm.normalized_log_p_vis(X))
tst_lp = gp.mean(rbm.normalized_log_p_vis(TX))

print "Average log p(x from training set) =  %f" % tr_lp
print "Average log p(x from test set) =      %f" % tst_lp
