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
iters = 20
k = 10
beta = 2

# dataset
by = ml.rbm.orrbm.base_y
overlaps = [(0, by), (5, by), (10, by), (15, by)]



