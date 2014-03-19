from __future__ import division

from ml.rbm.config import TrainingConfiguration
from math import sqrt
import ml.rbm.orrbm

# RBM configuration
n_vis=784
n_hid=5000
rbm_cfg = TrainingConfiguration(dataset='mnistv',
                                n_vis=n_vis, n_hid=n_hid,
                                batch_size=20,
                                n_gibbs_steps=15,
                                epochs=30,
                                step_rate=0.1,
                                use_pcd=True,
                                binarize_data='round',
                                initial_momentum=0, final_momentum=0, 
                                use_final_momentum_from_epoch=0,
                                weight_cost=0,
                                init_method='uniform', 
                                init_weight_sigma=4 * sqrt(6. / (n_hid + n_vis)), 
                                init_bias_sigma=0,
                                seed=1)

# Classifier
classifier='mlp'

# ORRBM
n_samples = 10000

# calculate cross-entropy
H_points = 2
H_iters = 10
H_k = 10
H_beta = 2

# separating
iters = 20
k = 10
beta = 2

# dataset
xrng = range(8, ml.rbm.orrbm.width)
shifts = [(x, ml.rbm.orrbm.base_y) for x in xrng]

# cross entropy calculation
test_x_shifts = xrng
test_y_shifts = [rbm.orrbm.base_y]




