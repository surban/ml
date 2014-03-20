
import numpy as np

import ml.datasets.skin
import ml.simple.table


def build_nextstep_data(ds, purpose, taxel):
    """
    Returns an array of inputs and an array of targets for next step prediction.
    :type ds: ml.datasets.skin.SkinDataset
    """
    X = np.zeros((2, 0))
    Z = np.zeros((0, ))
    for i in range(ds.record_count(purpose, taxel)):
        rec = ds.record(purpose, taxel, i)
        x = rec[:, 0:-1]
        z = rec[1, 1:]

        X = np.concatenate((X, x), axis=1)
        Z = np.concatenate((Z, z), axis=1)
    return X, Z




