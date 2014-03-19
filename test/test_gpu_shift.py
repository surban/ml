import numpy as np

import ml.nn.shift
import ml.datasets.shift_gpu
from ml.common.gpu import gather


def verify_shift_dataset(inputs, shifts, targets):
    #amounts = nn.shift.shift_amounts(shifts)
    for i in range(inputs.shape[1]):
        x = inputs[:,i]
        s = shifts[:,i]
        t = targets[:,i]
        xs = ml.nn.shift.shifted(x, s)

        print "x:       ", x
        print "shift:   ", s
        print "shifted: ", xs
        print "target:  ", t
        print

        if np.any(xs != t):
            print "Shift mismatch:"
            print "x:       ", x
            print "shift:   ", s
            print "shifted: ", xs
            print "target:  ", t
            return False
    return True
 

def test_generate_data():
    x_len = 20
    inputs, shifts, targets = ml.datasets.shift_gpu.generate_data(x_len, x_len, 10000)
    inputs = gather(inputs)
    shifts = gather(shifts)
    targets = gather(targets)
    assert verify_shift_dataset(inputs, shifts, targets), "data set verification failed"




def test_gpu_shift():
    x_len = 100
    s_len = x_len
    n_samples = 500000

    print "random data:"
    data = ml.datasets.shift_gpu.generate_random_data(x_len, n_samples, binary=True)
    print data

    shifts, shifts_hot = ml.datasets.shift_gpu.generate_shifts(x_len, n_samples)
    print "shifts:"
    print shifts
    print "shifts_hot:"
    print shifts_hot

    print "shifted:"
    shifted = ml.datasets.shift_gpu.generate_shifted(data, shifts)
    print shifted


