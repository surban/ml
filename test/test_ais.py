# -*- coding: utf-8 -*-

import scipy.io
import numpy as np
import gnumpy as gp
import time
import os

import rbm.rbm
import rbm.util
import rbm.ais


def test_compare_ais_with_ruslan():
    "Loads Ruslan's RBM and compares AIS results"
    ref_file = "test/rbm-for-ais-test.mat"
    epsilon = 0.2

    gp.seed_rand(int(time.time()))

    mdata = scipy.io.loadmat(ref_file)
    ref_logpf = mdata['logZZ_est'][0,0]
    ref_logpf_low = mdata['logZZ_est_down'][0,0]
    ref_logpf_high = mdata['logZZ_est_up'][0,0]
    n_hid = int(mdata['numhid'][0,0])

    X, TX = rbm.util.load_ruslan_mnist()
    myrbm = rbm.rbm.RestrictedBoltzmannMachine(100, 784, n_hid, 0)
    rbm.util.load_ruslan_parameters(myrbm, ref_file)
   
    ais = rbm.ais.AnnealedImportanceSampler(myrbm)
    ais.init_using_dataset(X)
    betas = np.concatenate((np.linspace(0.0, 0.5,   500, endpoint=False),
                            np.linspace(0.5, 0.9,  4000, endpoint=False),
                            np.linspace(0.9, 1.0, 10000)))
    logpf, logpf_low, logpf_high = ais.log_partition_function(betas=betas,
                                                              ais_runs=100)

    print "Test:      log Z = %g (%g, %g)" % (logpf, logpf_low, logpf_high)
    print "Reference: log Z = %g (%g, %g)" % (ref_logpf, ref_logpf_low, ref_logpf_high)
    assert abs(logpf - ref_logpf) < epsilon
    assert abs(logpf_low - ref_logpf_low) < epsilon
    assert abs(logpf_high - ref_logpf_high) < epsilon
