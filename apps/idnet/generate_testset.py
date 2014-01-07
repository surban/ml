import nn.id
import nn.gpushift
import os
import numpy as np
import apps.idnet.batchcfg as batchcfg

try:
    os.mkdir("testsets")
except:
    pass
os.chdir("testsets")

for x_len in batchcfg.x_lens:
    print "Generating x_len=%d" % x_len
    inputs, targets = nn.gpushift.generate_id_data(x_len, batchcfg.test_samples)
    np.savez_compressed("%d.npz" % x_len, inputs=inputs, targets=targets)




