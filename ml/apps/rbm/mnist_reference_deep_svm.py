import cPickle
import itertools
import gzip
import time

import numpy as np
import gnumpy as gp
import theano
import theano.tensor as T

import climin.stops
import climin.initialize
import climin.project
import climin.schedule

from brummlearn.pca import Pca
from brummlearn.mlp import Mlp, DropoutMlp, dropout_optimizer_conf, FastDropoutNetwork
from breze.component.loss import nce
import breze.util

import brummlearn.base
from brummlearn.data import one_hot

from climin.mathadapt import scalar
from ml.common.util import get_base_dir

savepath = "../mnist_deep_svm_model.npz"
datafile = get_base_dir() + "/datasets/mnist.pkl.gz"

# Load data.                                                                                                   
with gzip.open(datafile,'rb') as f:                                                                        
    train_set, val_set, test_set = cPickle.load(f)                                                       

X, Z = train_set                                                                                               
VX, VZ = val_set
TX, TZ = test_set

Z = one_hot(Z, 10)
VZ = one_hot(VZ, 10)
TZ = one_hot(TZ, 10)

Z = (Z * 2) - 1
TZ = (TZ * 2) - 1
VZ = (VZ * 2) - 1

image_dims = 28, 28

pca = Pca(70)
pca.fit(X)
X, VX, TX = [pca.transform(i) for i in (X, VX, TX)]

X = np.concatenate([X, VX], axis=0)
Z = np.concatenate([Z, VZ], axis=0)

X, Z, TX, TZ = [brummlearn.base.cast_array_to_local_type(i) for i in (X, Z, TX, TZ)]

# parameters
c_wd = 0.001
max_passes = 400
batch_size = 200
max_iter = max_passes * X.shape[0] / batch_size
n_report = X.shape[0] / batch_size



def squared_hinge(target, prediction):
    return (T.maximum(1 - target * prediction, 0) ** 2)

class TangMlp(Mlp):
    
    def __init__(
        self, n_inpt, n_hiddens, n_output, hidden_transfers, out_transfer, loss, optimizer, batch_size, noise_schedule,
        max_iter=1000, verbose=False):
        super(TangMlp, self).__init__(n_inpt, n_hiddens, n_output, hidden_transfers, out_transfer, loss, optimizer, batch_size,
            max_iter, verbose)
        
        self.noise_schedule = noise_schedule
    
    def _make_args(self, X, Z):
        args = super(TangMlp, self)._make_args(X, Z)
        def corrupt(x, level):
            return x + np.random.normal(0, level, x.shape).astype(theano.config.floatX)
        return (((corrupt(x, n), z), k) for n, ((x, z), k) in itertools.izip(self.noise_schedule, args))


def build_model():
    noise_schedule = (1 - float(i) / (max_iter+1) for i in xrange(max_iter)) 
    noise_schedule = itertools.chain(noise_schedule, itertools.repeat(1e-6))

    optimizer = 'rmsprop', {'steprate': 0.001, 'momentum': 0.9, 'decay': 0.9, 'step_adapt': 0}
    #optimizer = 'gd', {'steprate': climin.schedule.linear_annealing(0.1, 0, max_iter), 'momentum': 0.5, 'momentum_type': 'nesterov'}
    #optimizer = dropout_optimizer_conf(steprate_0=1, n_repeats=1)
    m = TangMlp(X.shape[1], [512, 512], 10, hidden_transfers=['sigmoid', 'sigmoid'], out_transfer='identity', loss=squared_hinge, noise_schedule=noise_schedule, optimizer=optimizer, batch_size=batch_size, max_iter=max_iter)
    climin.initialize.randomize_normal(m.parameters.data, 0, 0.02)
    m.parameters['out_bias'][...] = 0

    return m


def train_model():
    m = build_model()

    stop = climin.stops.any_([
        climin.stops.after_n_iterations(max_iter),
        ])
    pause = climin.stops.modulo_n_iterations(n_report)

    weight_decay = ((m.parameters.hidden_to_out ** 2).sum())
    #                + (m.parameters.hidden_to_hidden_0**2).sum()
    #                + (m.parameters.hidden_to_out**2).sum())
    weight_decay /= m.exprs['inpt'].shape[0]
    m.exprs['true_loss'] = m.exprs['loss']
    m.exprs['loss'] = m.exprs['loss'] + c_wd * weight_decay

    f_wd = m.function(['inpt'], c_wd * weight_decay)
    n_wrong = 1 - T.eq(T.argmax(m.exprs['output'], axis=1), T.argmax(m.exprs['target'], axis=1)).mean()
    f_n_wrong = m.function(['inpt', 'target'], n_wrong)
                
    losses = []
    v_losses = []
    print 'max iter', max_iter

    start = time.time()
    # Set up a nice printout.
    keys = '#', 'loss', 'val loss', 'seconds', 'wd', 'train emp', 'test emp'
    max_len = max(len(i) for i in keys)
    header = '\t'.join(i for i in keys)
    print header
    print '-' * len(header)

    f_loss = m.function(['inpt', 'target'], ['true_loss', 'loss'])

    for i, info in enumerate(m.powerfit((X, Z), (TX, TZ), stop, pause)):
        if info['n_iter'] % n_report != 0:
            continue
        passed = time.time() - start
        losses.append(info['loss'])
        v_losses.append(info['val_loss'])
    
        #img = tile_raster_images(fe.parameters['in_to_hidden'].T, image_dims, feature_dims, (1, 1))
        #save_and_display(img, 'filters-%i.png' % i)  
        info.update({
            'time': passed,
            'l2-loss': scalar(f_wd(X)),
            'train_emp': scalar(f_n_wrong(X, Z)),
            'test_emp': scalar(f_n_wrong(TX, TZ)),
        })
        row = '%(n_iter)i\t%(loss)g\t%(val_loss)g\t%(time)g\t%(l2-loss)g\t%(train_emp)g\t%(test_emp)g' % info
        print row


    np.savez_compressed(savepath, parameters=gp.as_numpy_array(m.parameters.data[...]))

    #m.f_loss = None
    #m.f_predict = None
    #with gzip.open(savepath, 'wb') as dumpfile:
    #    cPickle.dump(m, dumpfile)



def build_predictor():
    #with gzip.open(savepath, 'rb') as dumpfile:
    #    m = cPickle.load(dumpfile)

    m = build_model()
    data = np.load(savepath)
    m.parameters.data[...] = gp.as_garray(data['parameters'])

    def predict(x):
        x = np.asarray(x, dtype='float32')
        px = pca.transform(x)
        probs = gp.as_numpy_array(m.predict(px))
        return np.argmax(probs, axis=1)

    return predict

