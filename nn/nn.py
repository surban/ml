import theano
import theano.tensor as T


class OneHiddenLayerNN(object):
    """Feed-forward neural network with one hidden layer"""

    def __init__(self, W, Wb, V, Vb):
        self.W = W
        self.Wb = Wb
        self.V = V
        self.Vb = Vb

    def hid_act(self, vis):
        return T.nnet.sigmoid(T.dot(self.V, vis) + T.shape_padright(self.Vb))

    def out_act(self, hid):
        return T.dot(self.W, hid) + T.shape_padright(self.Wb)

    def regression_objective(self, inputs, targets):
        pred = self.out_act(self.hid_act(inputs))
        obj = T.mean((pred - targets)**2)
        return obj



class StackedRBF(object):
    """RBF network on top of neural network"""

    def __init__(self, P, W, V, l):
        self.P = P
        self.W = W
        self.V = V
        self.l = l

    def hid_act(self, vis):
        return T.nnet.sigmoid(T.dot(self.V, vis))

    def out_act_from_hid(self, hid):
        K = self.kernel_matrix(self.P, hid, self.l)
        return T.dot(self.W, K)

    def out_act(self, vis):
        return self.out_act_from_hid(self.hid_act(vis))

    @staticmethod
    def kernel_matrix(x, y, l):
        n_features = T.shape(x)[0]
        n_samples_x = T.shape(x)[1]
        n_samples_y = T.shape(y)[1]

        kx = T.reshape(x.T, (1, n_samples_x, n_features))
        kx = T.addbroadcast(kx, 0)
        
        ky = T.reshape(y.T, (n_samples_y, 1, n_features))
        ky = T.addbroadcast(ky, 1)

        d = T.sum((kx-ky)**2, 2, keepdims=False)
        K = T.exp(-d / (2*l**2))

        return K.T

    def regression_objective(self, inputs, targets):
        pred = self.out_act(inputs)
        obj = T.mean((pred - targets)**2)
        return obj





