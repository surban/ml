from __future__ import division

# workaround for Theano bug
# turn off for training reference_deep_svm
os.environ['BREZE_PARAMETERSET_DEVICE'] = 'cpu'

from ml.rbm.rbm import train_rbm, RestrictedBoltzmannMachine
import ml.rbm.util
import ml.rbm.sampling
import ml.rbm.accuracy
import ml.rbm.orrbm
import ml.common
import ml.common.util
import gnumpy as gp
import numpy as np
import os
import cPickle
import gc
import matplotlib.pyplot as plt

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


def plot_or_direct(OX, OY, n_samples):
    OX = OX[0:n_samples, :, :]
    OY = OY[0:n_samples, :, :]
    for n in range(OX.shape[0]):
        ax = plt.subplot(OX.shape[0], 2, 1+2*n)
        ml.common.util.ipy_plot_samples(OX[n,:,:], twod=True)
        ax.set_xticks([])
        ax.set_yticks([])

        ax = plt.subplot(OX.shape[0], 2, 1+2*n+1)
        ml.common.util.ipy_plot_samples(OY[n,:,:], twod=True)
        ax.set_xticks([])
        ax.set_yticks([])


# configuration
cfg, plot_dir = ml.common.util.standard_cfg()
os.chdir(plot_dir)
tcfg = cfg.rbm_cfg

print "Loading ml.rbm..."
try:
    epoch = tcfg.epochs - 1
    myrbm = RestrictedBoltzmannMachine(tcfg.batch_size, tcfg.n_vis, tcfg.n_hid, 
                                       tcfg.n_gibbs_steps)
    ml.rbm.util.load_parameters(myrbm, epoch)
except IOError, e:
    print "Failed: ", e
    print "Training ml.rbm..."
    ml.common.show_progress = True
    myrbm = train_rbm(tcfg, print_cost=False)
    ml.common.show_progress = False

print "Loading classifier..."
if cfg.classifier == 'mlp':
    import ml.apps.rbm.mnist_reference_dropout as ref
elif cfg.classifier == 'deepsvm':
    import ml.apps.rbm.mnist_reference_deep_svm as ref
else:
    assert False, "unknown classifier"

try:
    ref_predict = ref.build_predictor()
except IOError, e:
    print "Loading classifier failed: ", e
    print "Training classifier..."
    ref.train_model()
    ref_predict = ref.build_predictor()

print "Calculating classifier performance on single digits..."
ref_Z = ml.common.util.map(gp.as_numpy_array(tcfg.X), map_batch, ref_predict)
ref_TZ = ml.common.util.map(gp.as_numpy_array(tcfg.TX), map_batch, ref_predict)
ref_acc = 1 - ml.common.util.classification_error(ref_Z, tcfg.Z)
ref_tacc = 1 - ml.common.util.classification_error(ref_TZ, tcfg.TZ)
print "Classification error on whole training set:  %g" % (1-ref_acc)
print "Classification error on whole test set:      %g" % (1-ref_tacc)


# single digits (use test set)
S = tcfg.TX.reshape((tcfg.TX.shape[0], img_width, img_height))
SZ = tcfg.TZ

# sample indices for ORed data generation
sample_indices = ml.rbm.orrbm.generate_sample_indices_for_or_dataset(S, cfg.n_samples)

# calculate accuracies for different overlaps
orrbm_accs = {}
direct_accs = {}
for x_overlap, y_overlap in cfg.overlaps:
    print
    print "Overlap: x=%02d y=%02d" % (x_overlap, y_overlap)

    # calculate shift
    x_shift = img_width - x_overlap
    y_shift = ml.rbm.orrbm.base_y - y_overlap
    
    # generate dataset
    X, XZ, ref_XZ, Y, YZ, ref_YZ, O = \
        ml.rbm.orrbm.generate_or_dataset_with_shift(S, SZ, ref_Z,
                                                 x_shift, y_shift, cfg.n_samples,
                                                 sample_indices)   
       
    # calculate accuracy of applying classifier directly onto ORed digits
    OX = O[:,rbm.orrbm.base_y:rbm.orrbm.base_y+img_height,0:img_width]
    OY = O[:,y_shift:y_shift+img_height,x_shift:x_shift+img_width] 
    dacc = ml.rbm.accuracy.calc_separation_accuracy("(%02d, %02d) direct" % (x_overlap, y_overlap),
                                                 ref_predict, None,
                                                 X, XZ, ref_XZ,
                                                 Y, YZ, ref_YZ,
                                                 O,
                                                 OX, OY)       
    dacc.clear_stored_samples()
    direct_accs[(x_overlap, y_overlap)] = dacc
    print "Classification accuracy without separation:  %.4f (corrected: %.4f)" % (dacc.raw_accuracy, dacc.accuracy)

    # plot data for direct classification
    plt.figure(figsize=(3, 30))
    plot_or_direct(OX, OY, 10)
    plt.savefig("ordirect_%02d_%02d.png" % (x_overlap, y_overlap), dpi=300)

    plt.figure()
    ml.common.util.ipy_plot_samples(O[0:10,:,:], twod=True)
    plt.savefig("or_%02d_%02d.png" % (x_overlap, y_overlap), dpi=300)

    #continue

    # separate digits using ORRBM
    sep_XY = ml.common.util.map(O, map_batch,
                             lambda o: tuple_to_array(rbm.orrbm.or_infer_with_shift(myrbm, o, x_shift, y_shift, 
                                                                                    iters=cfg.iters, 
                                                                                    k=cfg.k, 
                                                                                    beta=cfg.beta)),
                             force_output_type='numpy',
                             caption="Separating digits with ORRBM")
    sep_X = sep_XY[:,:,:,0]
    sep_Y = sep_XY[:,:,:,1] 
    
    # calculate classification accuracy of digits separated by ORRBM
    oacc = ml.rbm.accuracy.calc_separation_accuracy("(%02d, %02d)" % (x_overlap, y_overlap),
                                                 ref_predict, myrbm,
                                                 X, XZ, ref_XZ,
                                                 Y, YZ, ref_YZ,
                                                 O,
                                                 sep_X, sep_Y)       
    oacc.clear_stored_samples()
    orrbm_accs[(x_overlap, y_overlap)] = oacc
    print "Classification accuracy after separation:    %.4f (corrected: %.4f)" % (oacc.raw_accuracy, oacc.accuracy)
    
    # cleanup to save memory
    del sep_X, sep_Y, sep_XY, X, XZ, ref_XZ, Y, YZ, ref_YZ, O, OX, OY
    gc.collect()
    gp.free_reuse_cache()
      
# save results
with open("accuracy.dat", mode='wb') as dump_file:
    cPickle.dump({'orrbm_accs': orrbm_accs,
                  'direct_accs': direct_accs}, 
                 dump_file, cPickle.HIGHEST_PROTOCOL)   
    
 