# -*- coding: utf-8 -*-

import scipy.io
import numpy as np
import gnumpy as gp
import time
import random

import rbm.rbm
import rbm.util
import rbm.ais
import rbm.config
import common
import common.util
import common.stats

def test_compare_rbm_with_ruslan():
    """Trains own RBM and compares average likelihood on training and test set 
    with RBM trained by Ruslan"""
    ref_file = "test/rbm-for-ais-test.mat"
    iterations = 10
    alpha = 0.10

    mdata = scipy.io.loadmat(ref_file)
    ref_logpf = mdata['logZZ_est'][0,0]
    ref_logpf_low = mdata['logZZ_est_down'][0,0]
    ref_logpf_high = mdata['logZZ_est_up'][0,0]
    ref_ll_training = mdata['loglik_training_est'][0,0]
    ref_ll_test = mdata['loglik_test_est'][0,0]
    n_hid = int(mdata['numhid'][0,0])
    cd = int(mdata['CD'][0,0])
    epochs = int(mdata['maxepoch'][0,0])

    ll_trainings = []
    ll_tests = []
    print "Running %d iterations" % iterations
    for i in range(iterations):
        tcfg = rbm.config.TrainingConfiguration(dataset='rmnist',
                                                n_vis=784, n_hid=n_hid,
                                                batch_size=100,
                                                n_gibbs_steps=cd,
                                                epochs=epochs,
                                                step_rate=0.05,
                                                use_pcd=False,
                                                binarize_data=True,
                                                initial_momentum=0.5, final_momentum=0.9, 
                                                use_final_momentum_from_epoch=5,
                                                weight_cost=0.0002,
                                                init_weight_sigma=0.01, init_bias_sigma=0,
                                                seed=random.randint(0, 100000))
        myrbm = rbm.rbm.train_rbm(tcfg)
   
        ais = rbm.ais.AnnealedImportanceSampler(myrbm)
        ais.init_using_dataset(tcfg.X)
        betas = np.concatenate((np.linspace(0.0, 0.5,   500, endpoint=False),
                                np.linspace(0.5, 0.9,  4000, endpoint=False),
                                np.linspace(0.9, 1.0, 10000)))
        logpf, logpf_low, logpf_high = ais.log_partition_function(betas=betas,
                                                                  ais_runs=100)
        myrbm.log_pf = logpf

        print "Test:      log Z = %g (%g, %g)" % (logpf, logpf_low, logpf_high)
        print "Reference: log Z = %g (%g, %g)" % (ref_logpf, ref_logpf_low, ref_logpf_high)

        ll_training = gp.mean(myrbm.normalized_log_p_vis(tcfg.X))
        ll_test = gp.mean(myrbm.normalized_log_p_vis(tcfg.TX))
        print "Test:      Average log p(x from training set) =  %f" % ll_training
        print "Test:      Average log p(x from test set)     =  %f" % ll_test

        ll_trainings.append(ll_training)
        ll_tests.append(ll_test)


    ll_training_mean, ll_training_pm = common.stats.normal_mean(ll_trainings, alpha)
    ll_test_mean, ll_test_pm = common.stats.normal_mean(ll_tests, alpha)
    print
    print "Reference: Average log p(x from training set)   =  %f" % ref_ll_training
    print "Test:      <Average log p(x from training set)> =  %f +/- %f" % \
        (ll_training_mean, ll_training_pm)
    print "Reference: Average log p(x from test set)       =  %f" % ref_ll_test
    print "Test:      <Average log p(x from test set)>     =  %f +/- %f" % \
        (ll_test_mean, ll_test_pm)

    assert common.util.interval_contains(common.stats.normal_mean_confint(ll_trainings, 
                                                                          alpha),
                                         ref_ll_training) or ll_training_mean > ref_ll_training
    assert common.util.interval_contains(common.stats.normal_mean_confint(ll_tests, 
                                                                          alpha),
                                         ref_ll_test) or ll_test_mean > ref_ll_test



def test_compare_rbm_with_lisa():
    """Trains own RBM and compares average likelihood on training and test set 
    with RBM trained by the Deep Learning Tutorials"""
    ref_file = "test/lisa-rbm.npz"
    iterations = 10
    #iterations = 2
    alpha = 0.10

    common.show_progress = True

    n_vis=784
    n_hid=500
    init_weight_sigma = 4 * np.sqrt(6. / (n_hid + n_vis))

    refrbm = rbm.rbm.RestrictedBoltzmannMachine(20, n_vis, n_hid, 0)
    rbm.util.load_parameters(refrbm, ref_file)

    lls = []
    ref_lls = []
    print "Running %d iterations" % iterations
    for i in range(iterations):
        tcfg = rbm.config.TrainingConfiguration(dataset='mnistv',
                                                n_vis=n_vis, n_hid=n_hid,
                                                batch_size=20,
                                                n_gibbs_steps=15,
                                                epochs=15,
                                                step_rate=0.1,
                                                use_pcd=True,
                                                binarize_data=False,
                                                initial_momentum=0, final_momentum=0, 
                                                use_final_momentum_from_epoch=0,
                                                weight_cost=0,
                                                init_method='uniform', init_weight_sigma=init_weight_sigma, init_bias_sigma=0,
                                                seed=random.randint(0, 100000))
        myrbm = rbm.rbm.train_rbm(tcfg)
   
        # AIS on my RBM
        ais = rbm.ais.AnnealedImportanceSampler(myrbm)
        ais.init_using_dataset(tcfg.X)
        betas = np.concatenate((np.linspace(0.0, 0.5,   500, endpoint=False),
                                np.linspace(0.5, 0.9,  4000, endpoint=False),
                                np.linspace(0.9, 1.0, 10000)))
        logpf, logpf_low, logpf_high = ais.log_partition_function(betas=betas,
                                                                  ais_runs=100)
        myrbm.log_pf = logpf
        print "Test:      log Z = %g (%g, %g)" % (logpf, logpf_low, logpf_high)

        # log likelihood of my RBM
        ll = gp.mean(myrbm.normalized_log_p_vis(tcfg.X))
        lls.append(ll)
        print "Test:      Average log p(x from training set) =  %f" % ll

        # AIS on reference RBM
        refais = rbm.ais.AnnealedImportanceSampler(refrbm)
        refais.init_using_dataset(tcfg.X)
        betas = np.concatenate((np.linspace(0.0, 0.5,   500, endpoint=False),
                                np.linspace(0.5, 0.9,  4000, endpoint=False),
                                np.linspace(0.9, 1.0, 10000)))
        logpf, logpf_low, logpf_high = ais.log_partition_function(betas=betas,
                                                                  ais_runs=100)
        refrbm.log_pf = logpf
        print "Reference: log Z = %g (%g, %g)" % (logpf, logpf_low, logpf_high)

        # log likelihood of my RBM
        ref_ll = gp.mean(refrbm.normalized_log_p_vis(tcfg.X))
        ref_lls.append(ref_ll)
        print "Test:      Average log p(x from training set) =  %f" % ref_ll
     
        # print statistics
        ll_mean, ll_pm = common.stats.normal_mean(lls, alpha)
        ref_ll_mean, ref_ll_pm = common.stats.normal_mean(ref_lls, alpha)
        print "############################################################"
        print "After %d iterations:" % i
        print "Test:      <Average log p(x from training set)> =  %g +/- %g" % \
            (ll_mean, ll_pm)
        print "Reference: <Average log p(x from training set)> =  %g +/- %g" % \
            (ref_ll_mean, ref_ll_pm)

    assert common.util.interval_contains(common.stats.normal_mean_confint(lls, 
                                                                          alpha),
                                         ref_ll_mean) or ll_mean > ref_ll_mean


