import numpy as np
import theano
import theano.tensor as T
import scipy.io
import climin

from operator import mul

def floatx(x):
    return np.asarray(x, dtype=theano.config.floatX)


class ParameterMerger(object):

    def __init__(self, *args):
        self.shapes = [arg.shape for arg in args]

        self.total_size = 0
        for s in self.shapes:
            l = reduce(mul, s)
            self.total_size += l

    def make_func(self, f):
        def wrapper(merged):
            args = self.unflatten(merged)
            ret = f(*args)
            if isinstance(ret, (list, tuple)):
                return self.flatten(*ret)
            elif ret.size > 1:
                return self.flatten(ret)
            else:
                return ret
            return self.flatten(*ret)
        return wrapper

    def flatten(self, *args):
        merged = np.zeros((self.total_size,))
        pos = 0
        for arg, s in zip(args, self.shapes):
            assert arg.shape == s, "argument shape %s does not equal %s" % \
                (arg.shape, s)
            l = reduce(mul, s)
            merged[pos : pos+l] = np.reshape(arg, (l,))
            pos += l
        return merged

    def unflatten(self, merged):
        pos = 0
        args = []
        for s in self.shapes:
            l = reduce(mul, s)
            arg = np.reshape(merged[pos : pos+l], s)
            args.append(arg)
            pos += l
        return tuple(args)



class SimpleNN(object):

    def __init__(self, W, V):
        self.W = W
        self.V = V

    def hiddens(self, vis):
        return T.nnet.sigmoid(T.dot(self.V, vis))

    def outputs(self, hid):
        return T.dot(self.W, hid)

    def objective(self, inputs, targets):
        pred = self.outputs(self.hiddens(inputs))
        obj = T.mean((pred - targets)**2)
        return obj


def load_data(filename):
    dat = scipy.io.loadmat(filename)
    RX = theano.shared(floatx(dat['RX']))
    RZ = theano.shared(floatx(dat['RZ']))
    VX = theano.shared(floatx(dat['VX']))
    VZ = theano.shared(floatx(dat['VZ']))
    TX = theano.shared(floatx(dat['TX']))
    TZ = theano.shared(floatx(dat['TZ']))
    return RX, RZ, VX, VZ, TX, TZ

if __name__ == '__main__':
    np.random.seed(100)
    RX, RZ, VX, VZ, TX, TZ = load_data('dat_boston.mat')

    n_targets = RZ.get_value().shape[0]
    n_features = RX.get_value().shape[0]
    n_hidden = 110

    # parameters
    W = floatx(np.random.random((n_targets, n_hidden)) - 0.5)
    V = floatx(np.random.random((n_hidden, n_features)) - 0.5)                            

    # theano variables
    inputs = T.matrix('inputs')
    targets = T.matrix('targets')
    TW = T.matrix('W')
    TV = T.matrix('V')

    # theano expressions
    nn = SimpleNN(TW, TV)
    RL = nn.objective(RX, RZ)
    VL = nn.objective(VX, VZ)
    TL = nn.objective(TX, TZ)
    dLdW = T.grad(RL, nn.W)
    dLdV = T.grad(RL, nn.V)

    # theano functions
    f_RL = theano.function(inputs=[TW, TV], outputs=RL)
    f_VL = theano.function(inputs=[TW, TV], outputs=VL)
    f_TL = theano.function(inputs=[TW, TV], outputs=TL)
    f_dL = theano.function(inputs=[TW, TV], outputs=[dLdW, dLdV])

    print "Error: %f" % f_RL(W, V)
    print "Error: %f" % f_VL(W, V)
    print "Error: %f" % f_TL(W, V)

    print "Gradient: ", f_dL(W, V)[0].shape, f_dL(W, V)[1].shape

    # optimizer
    pm = ParameterMerger(W, V)   
    par = pm.flatten()
    opt = climin.Lbfgs(par, pm.make_func(f_RL), pm.make_func(f_dL))

    # iterate
    for iter, sts in enumerate(opt):
        if iter % 100 == 0:
            W, V = pm.unflatten(par)
            print "%d: training: %f  validation: %f  test: %f" % \
                (iter, sts['loss'], f_VL(W,V), f_TL(W,V))
            




