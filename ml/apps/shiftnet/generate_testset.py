import os

import numpy as np

import ml.datasets.shift_gpu
import ml.apps.shiftnet.batchcfg as batchcfg
from ml.common.gpu import gather


try:
    os.mkdir("testsets")
except:
    pass
os.chdir("testsets")

for x_len in batchcfg.x_lens:
    print "Generating x_len=%d" % x_len
    inputs, shifts, targets = ml.datasets.shift_gpu.generate_data(x_len, x_len, batchcfg.test_samples)
    np.savez_compressed("%d.npz" % x_len,
                        inputs=gather(inputs),
                        shifts=gather(shifts),
                        targets=gather(targets))





