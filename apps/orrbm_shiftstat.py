from __future__ import division

from rbm.rbm import train_rbm, RestrictedBoltzmannMachine
from rbm.config import TrainingConfiguration
from rbm.ais import AnnealedImportanceSampler
from common.util import flatten_samples, unflatten_samples_like
import apps.mnist_reference_dropout as ref
import apps.generate_letters
import rbm.util
import rbm.sampling
import rbm.accuracy
import rbm.orrbm
import common
import common.util
import gnumpy as gp
import numpy as np
import pylab
import os
import pickle, cPickle
import gzip
import gc

#gp.expensive_check_probability = 0
#gp.track_memory_usage = True

# constants
img_width = 28
img_height = 28
map_batch = 1000


def tuple_to_array(t):
    x = t[0]
    y = t[1]
    out = np.zeros((x.shape[0], x.shape[1], x.shape[2], 2))
    out[:,:,:,0] = gp.as_numpy_array(x)
    out[:,:,:,1] = gp.as_numpy_array(y)
    return out    



# configuration
cfg, plot_dir = common.util.standard_cfg()
tcfg = cfg.rbm_cfg

print "Loading RBM..."
epoch = tcfg.epochs - 1
myrbm = RestrictedBoltzmannMachine(0, tcfg.n_vis, tcfg.n_hid, 0)
rbm.util.load_parameters(myrbm, epoch)

print "Loading classifier..."
if cfg.classifier == 'mlp':
    ref_mod = ref.build_model("../mnist_dropout_model.npz")
    ref_predict = ref.build_predictor(ref_mod)

    print "Calculating classifier performance on single digits..."
    ref_Z = common.util.map(gp.as_numpy_array(tcfg.X), map_batch, ref_predict)
    ref_TZ = common.util.map(gp.as_numpy_array(tcfg.TX), map_batch, ref_predict)
    ref_acc = 1 - common.util.classification_error(ref_Z, tcfg.Z)
    ref_tacc = 1 - common.util.classification_error(ref_TZ, tcfg.TZ)
    print "Classification error on whole training set:  %g" % (1-ref_acc)
    print "Classification error on whole test set:      %g" % (1-ref_tacc)
else:
    assert False, "unknown classifier"


# reshape single digits
S = tcfg.X.reshape((tcfg.X.shape[0], img_width, img_height))
SZ = tcfg.Z

# sample indices for ORed data generation
sample_indices = rbm.orrbm.generate_sample_indices_for_or_dataset(S, cfg.n_samples)

# calculate accuracies for different overlaps
orrbm_accs = {}
direct_accs = {}
for x_overlap, y_overlap in cfg.overlaps:
    print "Overlap: x=%02d y=%02d" % (x_overlap, y_overlap)

    # calculate shift
    x_shift = 28 - x_overlap
    y_shift = 28 - y_overlap
    
    # generate dataset
    X, XZ, ref_XZ, Y, YZ, ref_YZ, O = \
        rbm.orrbm.generate_or_dataset_with_shift(S, SZ, ref_Z,
                                                 x_shift, y_shift, cfg.n_samples,
                                                 sample_indices)   
       
    # calculate accuracy of applying classifier directly onto ORed digits
    OX = O[:,0:height,0:width]
    OY = O[:,y_shift:,x_shift:] 
    dacc = rbm.accuracy.calc_separation_accuracy("(%02d, %02d) direct" % (x_overlap, y_overlap), 
                                                 ref_predict, None,
                                                 X, XZ, ref_XZ,
                                                 Y, YZ, ref_YZ,
                                                 O,
                                                 OX, OY)       
    direct_accs[(x_overlap, y_overlap)] = dacc
    print "Direct classification accuracy:           %.3f" % dacc.accuracy

    # separate digits using ORRBM
    sep_XY = common.util.map(O, map_batch, 
                             lambda o: tuple_to_array(rbm.orrbm.or_infer_with_shift(myrbm, o, x_shift, y_shift, 
                                                                                    iters, k=k, beta=beta)),
                             force_output_type='numpy')
    sep_X = sep_XY[:,:,:,0]
    sep_Y = sep_XY[:,:,:,1] 
    
    # calculate classification accuracy of digits separated by ORRBM
    oacc = rbm.accuracy.calc_separation_accuracy("(%02d, %02d)" % (x_overlap, y_overlap), 
                                                 ref_predict, myrbm,
                                                 X, XZ, ref_XZ,
                                                 Y, YZ, ref_YZ,
                                                 O,
                                                 sep_X, sep_Y)       
    orrbm_accs[(x_overlap, y_overlap)] = oacc
    print "Classification accuracy after separation: %.3f" % oacc.accuracy
    
    # cleanup to save memory
    del sep_X, sep_Y, sep_XY, X, XZ, ref_XZ, Y, YZ, ref_YZ, O, OX, OY
    gc.collect()
    gp.free_reuse_cache()
      
# save results
with open("accuracy.dat", mode='wb') as dump_file:
    cPickle.dump({'orrbm_accs': orrbm_accs,
                  'direct_accs': direct_accs}, 
                 dump_file, cPickle.HIGHEST_PROTOCOL)   
    
 