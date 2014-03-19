import cPickle
import gzip
import time

import numpy as np
import theano.tensor as T

import climin.stops
import climin.initialize

from brummlearn.mlp import Mlp, DropoutMlp
from brummlearn.data import one_hot

from ml.common.util import get_base_dir

savepath = "../mnist_dropout_model.npz"
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

image_dims = 28, 28

def build_model(filename=None):
    max_passes = 100
    batch_size = 256
    max_iter = max_passes * X.shape[0] / batch_size
    n_report = X.shape[0] / batch_size

    stop = climin.stops.any_([
        climin.stops.after_n_iterations(max_iter),
        ])

    pause = climin.stops.modulo_n_iterations(n_report)

    optimizer = 'rmsprop', {'steprate': 0.0001, 'momentum': 0.95, 'decay': 0.9, 'step_adapt': 0.01}
    m = DropoutMlp(784, [800, 800], 10, hidden_transfers=['sigmoid', 'sigmoid'], out_transfer='softmax', loss='nce', optimizer=optimizer, batch_size=batch_size, max_iter=max_iter)
    m.parameters.data[...] = np.random.normal(0, 1e-1, m.parameters.data.shape)

    weight_decay = ((m.parameters.in_to_hidden**2).sum()
                    + (m.parameters.hidden_to_hidden_0**2).sum()
                    + (m.parameters.hidden_to_out**2).sum())
    weight_decay /= m.exprs['inpt'].shape[0]
    m.exprs['true_loss'] = m.exprs['loss']
    c_wd = 0.0000
    m.exprs['loss'] = m.exprs['loss'] + c_wd * weight_decay

    f_wd = m.function(['inpt'], c_wd * weight_decay)
                
    losses = []
    v_losses = []
    #print 'max iter', max_iter

    if filename is not None:
        data = np.load(filename)
        m.parameters.data[...] = data['parameters']

    return m

def save_model(m, filename):
    np.savez_compressed(filename, parameters=m.parameters.data[...])

def do_training(m):
    start = time.time()
    # Set up a nice printout.
    keys = '#', 'loss', 'val loss', 'seconds', 'wd'
    max_len = max(len(i) for i in keys)
    header = '   '.join(i.ljust(max_len) for i in keys)
    print header
    print '-' * len(header)

    f_loss = m.function(['inpt', 'target'], ['true_loss', 'loss'])

    for i, info in enumerate(m.powerfit((X, Z), (VX, VZ), stop, pause)):
        passed = time.time() - start
        losses.append(info['loss'])
        v_losses.append(info['val_loss'])
    
        #img = tile_raster_images(fe.parameters['in_to_hidden'].T, image_dims, feature_dims, (1, 1))
        #save_and_display(img, 'filters-%i.png' % i)  
    
        row = '%i' % info['n_iter'], '%.6f' % info['loss'], '%.6f' % info['val_loss'], '%.3f' % passed, str(f_wd(X))
        print '   '.join(i.ljust(max_len) for i in row)


def train_model():
    m = build_model()
    do_training(m)
    save_model(m, savepath)


def build_predictor():
    m = build_model(savepath)

    f_predict = m.function(['inpt'], T.argmax(m.exprs['output_in'], axis=1))
    return lambda x: f_predict(np.asarray(x, dtype='float32'))

