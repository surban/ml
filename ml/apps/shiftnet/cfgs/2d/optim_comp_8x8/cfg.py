# RUNNER:apps.shiftnet.shiftnet_2d

arch = 'comp'
width = 8
height = 8
n_val_samples = 1000
n_batch = 1000
optimizer = 'rmsprop'
init = 0.1
momentum = 0.8
new_data_iters = 1000
max_iters = 0
min_iters = None
start_with_optimal_weights = True

steprate_itr = [0,    9000]
steprate_val = [1e-4, 1e-5]

