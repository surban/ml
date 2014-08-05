# RUNNER:apps.shiftnet.shiftnet_2d

arch = 'comp'
width = 4
height = 4
n_val_samples = 1000
n_batch = 1000
optimizer = 'rmsprop'
init = 0.1
momentum = 0.8
new_data_iters = 1000
max_iters = None
min_iters = 20000

steprate_itr = [0,    9000]
steprate_val = [1e-4, 1e-5]

