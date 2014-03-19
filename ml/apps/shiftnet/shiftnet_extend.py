# -*- coding: utf-8 -*-

import ml.common.gpu
import ml.common.util

from math import isnan

import gnumpy as gp
import breze.util
import matplotlib.pyplot as plt
from scipy.linalg import block_diag

from ml.nn.shift import FourierShiftNet
from ml.common.complex import *
from ml.common.gpu import gather, post, function
from ml.datasets.shift import generate_data
import climin


np.set_printoptions(precision=3, suppress=True)


# hyperparameters
use_base_data = True
show_gradient = False
check_nans = False
cfg, plot_dir = ml.common.util.standard_cfg(prepend_scriptname=False)
cfg.steprate = ml.common.util.ValueIter(cfg.steprate_itr, cfg.steprate_val,
                                     transition='linear', transition_length=1000)

# load base parameters
lcls = {}
execfile(cfg.base_dir + "/cfg.py", {}, lcls)
base_x_len = lcls['x_len']
base_s_len = lcls['s_len']
bps = breze.util.ParameterSet(**FourierShiftNet.parameter_shapes(base_x_len, base_s_len))
bps.data[:] = post(np.load(cfg.base_dir + "/result.npz")['ps'])
assert cfg.x_len == 2*base_x_len and cfg.s_len == 2*base_s_len, "can only double neurons"

# test:
#cfg.x_len = base_x_len
#cfg.s_len = base_s_len

# parameters
ps = breze.util.ParameterSet(**FourierShiftNet.parameter_shapes(cfg.x_len, cfg.s_len))

# transfer base parameters so that nets are equivalent
def doubling_matrix(n):
    d = [[1, 1]]
    nd = [d for _ in range(n)]
    return block_diag(*nd)

def shift_doubling_matrix(n):
    d = [[1, 0]]
    nd = [d for _ in range(n)]
    return block_diag(*nd)

def double_weights(w):
    return 0.5 * np.dot(np.dot(doubling_matrix(w.shape[0]).T, w), doubling_matrix(w.shape[1]))

for wn in ps.views.iterkeys():
    ps[wn][:] = post(double_weights(gather(bps[wn])))
    # test:
    #ps[wn][:] = post(gather(bps[wn]))

ps['s_to_shat_re'][:] = 2.0 * ps['s_to_shat_re'][:]
ps['s_to_shat_im'][:] = 2.0 * ps['s_to_shat_im'][:]

# data generation
def generate_data(n_samples):
    if use_base_data:
        inputs, shifts, targets = generate_base_data(n_samples)
    else:
        inputs, shifts, targets = generate_data(cfg.x_len, cfg.s_len, n_samples)
    return inputs, shifts, targets

def generate_base_data(n_samples):
    inputs, shifts, targets = generate_data(base_x_len, base_s_len, n_samples)
    inputs = gp.dot(doubling_matrix(base_x_len).T, inputs)
    targets = gp.dot(doubling_matrix(base_x_len).T, targets)
    shifts = gp.dot(shift_doubling_matrix(base_s_len).T, shifts)
    return inputs, shifts, targets


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

# separate gradients wrt layer weights
if show_gradient:
    f_grads = {}
    for wname, wvar in ps.vars.iteritems():
        f_grads[wname] = function(inputs=[ps.flat,x,s,t], outputs=T.grad(loss, wvar))

print "Generating validation data..."
val_inputs, val_shifts, val_targets = generate_data(cfg.n_val_samples)
tst_inputs, tst_shifts, tst_targets = generate_data(cfg.n_val_samples)
print "Done."

# data generation debug
#print "val_inputs:"
#print val_inputs
#print "val_shifts:"
#print val_shifts
#print "val_targets:"
#print val_targets
#sys.exit(0)
                                    
# optimizer
def generate_new_training_data():
    global trn_inputs, trn_shifts, trn_targets
    print "Generating new data..."
    trn_inputs, trn_shifts, trn_targets = generate_data(cfg.n_batch)

def f_trn_loss(p):
    global trn_inputs, trn_shifts, trn_targets
    if check_nans:
        assert np.all(np.isfinite(gather(p))), "NaN in p given to f_trn_loss"
    return f_loss(p, trn_inputs, trn_shifts, trn_targets) 

def f_trn_dloss(p):
    global trn_inputs, trn_shifts, trn_targets
    if check_nans:
        assert np.all(np.isfinite(gather(p))), "NaN in p given to f_trn_dloss"
    dloss = f_dloss(p, trn_inputs, trn_shifts, trn_targets) 
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

#  base loss
global_ts = np.load("testsets/%d.npz" % cfg.x_len)
global_loss = gather(f_loss(ps.data, post(global_ts['inputs']), post(global_ts['shifts']), post(global_ts['targets'])))
print "Base loss on global test set: %f" % global_loss

# initial data
generate_new_training_data()

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
        use_base_data = not use_base_data
        generate_new_training_data()

    #if iter % 2000 == 0:
    #    opt.reset()

    if iter % 10 == 0:
        opt.steprate = cfg.steprate[iter]
        #print "steprate: ", cfg.steprate[iter]

        trn_loss = gather(f_trn_loss(ps.data))
        val_loss = gather(f_loss(ps.data, val_inputs, val_shifts, val_targets))
        tst_loss = gather(f_loss(ps.data, tst_inputs, tst_shifts, tst_targets))

        # plot histrogram of gradients
        if show_gradient:
            #plt.figure()
            print
            for p, wname in enumerate(sorted(f_grads.keys())):
                plt.subplot(5,2,p)
                fg = f_grads[wname]
                grad = gather(fg(ps.data, trn_inputs, trn_shifts, trn_targets))
                print "%s: |grad| = %e" % (wname, np.mean(np.abs(grad)))

                #plt.hist(grad)
                #plt.title(wname)
                #plt.savefig(plot_dir + "/grad_%04d.pdf" % iter)

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

# test on global test set
global_loss = gather(f_loss(ps.data, post(global_ts['inputs']), post(global_ts['shifts']), post(global_ts['targets'])))
print "Loss on global test set: %f" % global_loss

# save result
np.savez_compressed(plot_dir + "/result.npz",
                    ps=gather(ps.data), global_loss=global_loss, history=his.history)

