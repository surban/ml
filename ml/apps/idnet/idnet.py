# -*- coding: utf-8 -*-

import ml.common.plot
import ml.common.gpu

from math import isnan

import numpy as np
import theano.tensor as T
import breze.util
import matplotlib.pyplot as plt

import ml.common.util
import ml.nn.id
from ml.common.gpu import gather, post, function
from ml.apps.idnet.idnet_plot import plot_all_weights
from ml.datasets.shift import generate_id_data
import climin


# hyperparameters
check_nans = False
show_gradient = False
cfg, plot_dir = ml.common.util.standard_cfg(prepend_scriptname=False)
cfg.steprate = ml.common.util.ValueIter(cfg.steprate_itr, cfg.steprate_val,
                                        transition='linear', transition_length=1000)
if 'do_weight_plots' in dir(cfg):
    do_weight_plots = cfg.do_weight_plots
else:
    do_weight_plots = True

# parameters
ps = breze.util.ParameterSet(**ml.nn.id.FourierIdNet.parameter_shapes(cfg.x_len))

# inputs
x = T.matrix('x')
t = T.matrix('t')

# functions
fsn = ml.nn.id.FourierIdNet(**ps.vars)
out_re, out_im = fsn.output(x, output_y_im=True)

pure_loss = T.mean((out_re - t)**2)

loss = pure_loss
if 'train_im_zero' in dir(cfg) and cfg.train_im_zero:
    print "Including Im=0 in loss."
    loss += T.mean(out_im**2)
if 'mult_sparsity' in dir(cfg) and cfg.mult_sparsity > 0:
    print "Adding sparsity constraint on multiplicative weights with factor %e to loss" % cfg.mult_sparsity
    loss += cfg.mult_sparsity * T.sum(T.abs_(ps.Xhat_to_Yhat_re) + T.abs_(ps.Xhat_to_Yhat_im))
if 'penalize_small_yhat_to_y' in dir(cfg) and cfg.penalize_small_yhat_to_y > 0:
    print "Penalizing small yhat_to_y weights with factor %e" % cfg.penalize_small_yhat_to_y
    loss += cfg.penalize_small_yhat_to_y * T.sum(T.sqrt(T.sqr(ps.yhat_to_y_re) + T.sqr(ps.yhat_to_y_im) + 0.001)**(-4))
if 'penalize_small_x_to_xhat' in dir(cfg) and cfg.penalize_small_x_to_xhat > 0:
    print "Penalizing small x_to_xhat weights with factor %e" % cfg.penalize_small_x_to_xhat
    loss += cfg.penalize_small_x_to_xhat * T.sum(T.sqrt(T.sqr(ps.x_to_xhat_re) + T.sqr(ps.x_to_xhat_im) + 0.001)**(-4))
if 'tight_weights' in dir(cfg) and cfg.tight_weights:
    print "Tighing input and output weights"
    loss += T.sum((ps.x_to_xhat_re.T - ps.yhat_to_y_re)**2 + (ps.x_to_xhat_im.T + ps.yhat_to_y_im)**2)

f_output = function(inputs=[ps.flat, x], outputs=out_re)
f_loss = function(inputs=[ps.flat, x, t], outputs=loss)
f_pure_loss = function(inputs=[ps.flat, x, t], outputs=pure_loss)
f_dloss = function(inputs=[ps.flat, x, t], outputs=T.grad(loss, ps.flat))

# separate gradients wrt layer weights
if show_gradient:
    f_grads = {}
    for wname, wvar in ps.vars.iteritems():
        f_grads[wname] = function(inputs=[ps.flat, x, t], outputs=T.grad(loss, wvar))

if do_weight_plots:
    plt.figure()

print "Generating validation data..."
val_inputs, val_targets = generate_id_data(cfg.x_len, cfg.n_val_samples)
tst_inputs, tst_targets = generate_id_data(cfg.x_len, cfg.n_val_samples)
print "Done."
                                    
# optimizer
def generate_new_data():
    global trn_inputs, trn_targets
    #print "Generating new data..."
    trn_inputs, trn_targets = \
        generate_id_data(cfg.x_len, cfg.n_batch)

def f_trn_loss(p):
    global trn_inputs, trn_targets
    if check_nans:
        assert np.all(np.isfinite(gather(p))), "NaN in p given to f_trn_loss"
    return f_loss(p, trn_inputs, trn_targets)

def f_trn_dloss(p):
    global trn_inputs, trn_targets
    if check_nans:
        assert np.all(np.isfinite(gather(p))), "NaN in p given to f_trn_dloss"
    dloss = f_dloss(p, trn_inputs, trn_targets)
    if check_nans:
        if not np.all(np.isfinite(gather(dloss))):
            print "NaN in calcuated gradient"
            import pdb; pdb.set_trace()  
    return dloss

if cfg.optimizer == 'rprop':
    opt = climin.Rprop(ps.data, f_trn_dloss, 
                       step_shrink=cfg.step_shrink, max_step=cfg.max_step)
elif cfg.optimizer == 'rmsprop':
    opt = climin.RmsProp(ps.data, f_trn_dloss, 
                         steprate=cfg.steprate[0], 
                         momentum=cfg.momentum)
elif cfg.optimizer == 'gradientdescent':
    opt = climin.GradientDescent(ps.data, f_trn_dloss, 
                                 steprate=cfg.steprate[0],
                                 momentum=cfg.momentum)
else:
    assert False, "unknown optimizer"

# initialize parameters
if 'continue_training' in dir(cfg) and cfg.continue_training:
    print "Loading weights..."
    ps.data[:] = post(np.load(plot_dir + "/base.npz")['ps'])
elif 'start_with_optimal_weights' in dir(cfg) and cfg.start_with_optimal_weights:
    print "Initializing weights optimally..."
    res = ml.nn.id.FourierIdNet.optimal_weights(cfg.x_len)
    res = [post(x) for x in res]
    (ps['x_to_xhat_re'], ps['x_to_xhat_im'],
     ps['Xhat_to_Yhat_re'], ps['Xhat_to_Yhat_im'],
     ps['yhat_to_y_re'], ps['yhat_to_y_im']) = res
else:
    print "Initializing weights randomly..."
    ps.data[:] = cfg.init * (np.random.random(ps.data.shape) - 0.5)

if 'perturb_weights' in dir(cfg):
    for wname, scale in cfg.perturb_weights.iteritems():
        if scale > 0:
            print "Perturbing %s with sigma=%.3f" % (wname, scale)
            ps[wname] += np.random.normal(scale=scale, size=ps[wname].shape)

if 'randomize_weights' in dir(cfg):
    for wname in cfg.randomize_weights:
        print "Randomizing %s" % wname
        ps[wname] = cfg.init * (np.random.random(ps[wname].shape) - 0.5)

if 'zero_weights' in dir(cfg):
    for wname in cfg.zero_weights:
        print "Zeroing %s" % wname
        ps[wname] = 0

print "initial validation loss: ", f_loss(ps.data, val_inputs, val_targets)

# initial data
generate_new_data()

# optimize
his = ml.common.util.ParameterHistory(max_missed_val_improvements=1000,
                                      desired_loss=0.0001,
                                      max_iters=cfg.max_iters,
                                      min_iters=cfg.min_iters)
for iter, sts in enumerate(opt):
    if check_nans:
        assert np.all(np.isfinite(gather(sts['step']))), 'NaN in step'
        assert np.all(np.isfinite(gather(sts['moving_mean_squared']))), 'NaN in moving_mean_squared'
        assert np.all(np.isfinite(gather(ps.data))), 'NaN in ps.data'

    if iter % cfg.new_data_iters == 0:
        generate_new_data()

    #if iter % 1000 == 0:
    #    opt.reset()

    if iter % 10 == 0:
        opt.steprate = cfg.steprate[iter]
        #print "steprate: ", cfg.steprate[iter]

        trn_loss = gather(f_trn_loss(ps.data))
        val_loss = gather(f_pure_loss(ps.data, val_inputs, val_targets))
        tst_loss = gather(f_pure_loss(ps.data, tst_inputs, tst_targets))

        # plot histrogram of gradients
        if show_gradient:
            #plt.figure()
            print
            for p, wname in enumerate(sorted(f_grads.keys())):
                plt.subplot(5,2,p)
                fg = f_grads[wname]
                grad = gather(fg(ps.data, trn_inputs, trn_targets))
                print "%s: |grad| = %e" % (wname, np.mean(np.abs(grad)))

                #plt.hist(grad)
                #plt.title(wname)
                #plt.savefig(plot_dir + "/grad_%04d.pdf" % iter)

        # plot weights
        if iter % 500 == 0 and do_weight_plots:
            plt.clf()
            plot_all_weights(ps)
            plt.title("iter=%d" % iter)
            plt.draw()

            if ml.common.plot.headless:
                plt.savefig(plot_dir + "/weights_%d.pdf" % iter)
            else:
                plt.pause(0.05)

        if isnan(val_loss) or isnan(tst_loss):
            print "Encountered NaN, restoring parameters."
            import pdb; pdb.set_trace()            
            ps.data[:] = his.best_pars
            continue

        his.add(iter, ps.data, trn_loss, val_loss, tst_loss)
        if his.should_terminate:
            break
                      
ps.data[:] = his.best_pars
his.plot()
plt.savefig(plot_dir + "/loss.pdf")

# check with simple patterns
sim_inputs, sim_targets = generate_id_data(cfg.x_len, 3, binary=True)
sim_results = gather(f_output(ps.data, post(sim_inputs)))
print "input:   "
print sim_inputs.T
print "targets: "
print sim_targets.T
print "results: "
print sim_results.T

# test on global test set
try:
    global_ts = np.load("testsets/%d.npz" % cfg.x_len)
    global_loss = gather(f_pure_loss(ps.data, post(global_ts['inputs']), post(global_ts['targets'])))
    print "Loss on global test set after %d iterations: %g" % (his.performed_iterations, global_loss)
except:
    print "Finished after %d iterations" % his.performed_iterations
    global_loss = 0

# save results
np.savez_compressed(plot_dir + "/result.npz",
                    ps=gather(ps.data), global_loss=global_loss, history=his.history,
                    performed_iterations=his.performed_iterations)




