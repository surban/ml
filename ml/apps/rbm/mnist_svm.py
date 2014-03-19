# -*- coding: utf-8 -*-

import gnumpy as gp
import numpy as np
import pickle
import sys
import gzip

from sklearn import svm

import ml.common.util
import ml.rbm.util

def train():
    X, TX, y, Ty = ml.rbm.util.load_mnist(False)
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
    with gzip.open(filename, 'wb') as file:
        pickle.dump(svc, file, pickle.HIGHEST_PROTOCOL)

    return svc

def check_performance(svc):
    X, TX, y, Ty = ml.rbm.util.load_mnist(False)
    X = gp.as_numpy_array(X)
    y = gp.as_numpy_array(y)
    TX = gp.as_numpy_array(TX)
    Ty = gp.as_numpy_array(Ty)

    print "Checking performance..."
    nt = 10000
    Py = svc.predict(X[0:nt])
    training_err = ml.common.util.classification_error(Py, y[0:nt])
    PTy = svc.predict(TX)
    test_err = ml.common.util.classification_error(PTy, Ty)

    print "Prediction error on first %d training samples: %g" % (nt, training_err)
    print "Prediction error on test set:                  %g" % test_err

    return svc

def load():
    with gzip.open("mnist_svm.dat", 'rb') as file:
        return pickle.load(file)

if __name__ == '__main__':
    if sys.argv[1] == "load":
        svc = load()
    elif sys.argv[1] == "train":
        svc = train()
    check_performance(svc)

    


