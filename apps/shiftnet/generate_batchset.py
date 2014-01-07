import apps.shiftnet.batchcfg
import os

tmpl = """
x_len = %(x_len)d
s_len = x_len
n_val_samples = 10000
n_batch = 10000
optimizer = '%(optimizer)s'
init = 0.01
momentum = 0.9
new_data_iters = 1000
max_iters = None
min_iters = 8000

steprate_itr = None
steprate_val = 1e-4
"""

#optimizers = ['rprop', 'rmsprop']
optimizers = ['rmsprop']

i = 0
for optimizer in optimizers:
    for x_len in apps.shiftnet.batchcfg.x_lens:
        i += 1
        dirname = "batch_%04d" % i
        try:
            os.mkdir(dirname)
        except:
            pass

        with open(dirname + "/cfg.py", 'w') as cfgfile:
            cfg = tmpl % {'optimizer': optimizer, 'x_len': x_len}
            cfgfile.write(cfg)
        with open(dirname + "/CMD", 'w') as cmdfile:
            cmdfile.write("python shiftnet_mb.py %s" % dirname)




