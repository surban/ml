from sklearn.neighbors import KernelDensity

import numpy as np

kde = KernelDensity(kernel='gaussian', bandwidth=1.0)

si = np.random.random(size=(13074, 2))
print "fit"
kde.fit(si)

x = np.random.random(size=(27264, 2))
print "estimate"
kde.score_samples(x)


