import numpy as np
import cvxopt.solvers as solvers
import scipy.optimize as optimize
import scipy
import scipy.special._logsumexp as log
import pandas as pd
import A3helpers

solvers.options['show_progress'] = False

def mul_diviance(W, X, Y, d, k):
    W_reshape = W.reshape(d, k)
    term1 = X @ W_reshape # k x n
    term2 = log.logsumexp(term1, axis=1) 
    term3 = Y.T @ term1 
    term4 = np.diag(term3)

    return np.sum(term2) - np.sum(term4)


def minMulDev(X, Y):
    n, d = X.shape
    k = Y.shape[1]
    W = np.zeros(d * k)
    res = optimize.minimize(mul_diviance, W, args=(X, Y, d, k))
    # print(res)
    # print((d, k))
    return res.x.reshape((d, k))

#b
def indmax(X, k):
    y = np.argmax(X, axis=1)
    Y = np.empty((0, k))
    for row in y:
        Y = np.vstack([Y, A3helpers.convertToOneHot(row, k)])

    return Y

def classify(Xtest, W):
    #print(Xtest.shape, W.shape)
    term1 = Xtest @ W
    Yhat = indmax(term1, W.shape[1])
    return Yhat

def calculateAcc(Yhat, Y):
    predicted = np.argmax(Yhat, axis=1)
    ground = np.argmax(Y, axis= 1)
    # print(Yhat)
    # print(Y)
    return np.mean(predicted == ground)

def synRegExperiment():
    return A3helpers.synClsExperiments(minMulDev, classify, calculateAcc)
    
print(synRegExperiment())