import os

import numpy as np

import datasets.shift_gpu
import apps.shiftnet.batchcfg as batchcfg
from common.gpu import gather


try:
    os.mkdir("testsets")
except:
    pass
os.chdir("testsets")

for x_len in batchcfg.x_lens:
    print "Generating x_len=%d" % x_len
    inputs, shifts, targets = datasets.shift_gpu.generate_data(x_len, x_len, batchcfg.test_samples)
    np.savez_compressed("%d.npz" % x_len,
                        inputs=gather(inputs),
                        shifts=gather(shifts),
                        targets=gather(targets))





