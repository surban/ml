
import numpy as np
import matplotlib.pyplot as plt

from ml.apps.skin.autoregr_table import build_nextstep_data
from ml.simple.table import TableRegression
from ml.datasets.skin import SkinDataset


ds = SkinDataset("raising_small")
ds.print_statistics()


taxel = (1,1)

X, Z = build_nextstep_data(ds, 'train', taxel, n_curves=2)
print "Dataset size: ", X.shape[1]


force_min = 0
force_step = 0.1
force_max = 25
skin_min = 0
skin_step = 0.02
skin_max = 2

tr = TableRegression([force_min, skin_min],
                     [force_step, skin_step],
                     [force_max, skin_max])
print "Number of weights: %d" % tr._n_weights
tr.train(X, Z)
Zp = tr.predict(X)

err = np.mean((Z - Zp)**2)
print "One step prediction sqrt(MSE): ", err

