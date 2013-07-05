import theano
import theano.tensor as T

class ShiftNet(object):

    def __init__(self, WP, WS, WSbias, WL, WLbias, WO):
        self.WP = WP
        self.WS = WS
        self.WSbias = WSbias
        self.WL = WL
        self.WLbias = WLbias
        self.WO = WO


    def output(self, input, shift):
        # prelog layer
        pl = self.WP * input

        # log in layer
        lli = T.log(pl)

        # shift transform layer
        st = T.nnet.sigmoid(T.dot(self.WS, shift) + self.WSbias)

        # log out layer
        llo = T.dot(self.WL, lli) + self.WLbias

        # exp layer
        el = T.exp(llo)

        # output layer
        ol = self.WO * el


