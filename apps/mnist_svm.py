# -*- coding: utf-8 -*-

import gnumpy as gp
import numpy as np
import pickle
import sys

from sklearn import svm

import common.util
import rbm.util

def train():
    X, TX, y, Ty = rbm.util.load_mnist(False)
    X = gp.as_numpy_array(X)
    y = gp.as_numpy_array(y)
    TX = gp.as_numpy_array(TX)
    Ty = gp.as_numpy_array(Ty)

    #X = X[0:3000, ...]
    #y = y[0:3000, ...]

    print "Fitting SVM..."
    svc = svm.SVC(kernel='rbf', verbose=True)
    svc.fit(X, y)

    filename = "mnist_svm.dat"
    print "Writing model to %s" % filename
    with open(filename, 'wb') as file:
        pickle.dump(svc, file)

def check_performance(svc):
    X, TX, y, Ty = rbm.util.load_mnist(False)
    X = gp.as_numpy_array(X)
    y = gp.as_numpy_array(y)
    TX = gp.as_numpy_array(TX)
    Ty = gp.as_numpy_array(Ty)

    print "Checking performance..."
    #Py = svc.predict(X)
    #training_err = common.util.classification_error(Py, y)
    PTy = svc.predict(TX)
    test_err = common.util.classification_error(PTy, Ty)

    #print "Prediction error on training set: %g" % training_err
    print "Prediction error on test set:     %g" % test_err

    return svc

def load():
    with open("mnist_svm.dat", 'rb') as file:
        return pickle.load(file)

if __name__ == '__main__':
    if sys.argv[1] == "load":
        svc = load()
    else:
        svc = train()
    check_performance(svc)

    


