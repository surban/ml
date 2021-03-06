import numpy as np
import gnumpy as gp
import theano.tensor as T

from ml.nn.nn import StackedRBF
from ml.common.util import floatx
from ml.common.gpu import function


# load data
#np.random.seed(100)
#RX, RZ, VX, VZ, TX, TZ = ml.common.util.load_theano_data('../datasets/boston_split.mat')
#RX, RZ, VX, VZ, TX, TZ = ml.common.util.load_theano_data('../datasets/abalone_split.mat')

# check kernel
x=floatx(np.array([[11, 21, 31], [12, 22, 32]]))
y=floatx(np.array([[101, 201],   [102, 202]]))
x=gp.as_garray(x)
y=gp.as_garray(y)
l = gp.as_garray([[100]])


tx = T.matrix('x')
ty = T.matrix('y')
tl = T.matrix('l')
f_kernel_matrix = function([tx, ty, tl], StackedRBF.kernel_matrix(tx, ty, tl))


K = f_kernel_matrix(x, y, l)
print K


tsq = T.sum(tx**2)
f_sq = function([tx], tsq)
print f_sq(gp.as_garray([[5]]))


gsq = T.grad(tsq, [tx])
f_gsq = function([tx], gsq)
print f_gsq(gp.as_garray([[5]]))






