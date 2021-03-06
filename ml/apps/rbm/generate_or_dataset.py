# -*- coding: utf-8 -*-

import gnumpy as gp
import numpy as np

import ml.rbm.orrbm
import ml.rbm.util

samples = 10000
test_samples = 10000
seed = 500


X, TX, Z, TZ = ml.rbm.util.load_mnist(False)

np.random.seed(seed)

O, OZ = ml.rbm.orrbm.generate_or_dataset(X, Z, samples)
rbm.orrbm.save_or_dataset("ordata.npz", O, OZ)

TO, TOZ = ml.rbm.orrbm.generate_or_dataset(TX, TZ, test_samples)
rbm.orrbm.save_or_dataset("ordata_test.npz", TO, TOZ)



