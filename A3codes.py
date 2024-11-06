import numpy as np
import cvxopt.solvers as solvers
import scipy.optimize as optimize
import scipy
import scipy.special._logsumexp as log
import pandas as pd
import A3helpers

solvers.options['show_progress'] = False

def mul_diviance(W, X, Y, d, k):
    W_reshape = np.reshape(W, (d, k))
    term1 = X @ W_reshape
    term2 = log.logsumexp(term1, axis=1) 
    term3 = term1 @ Y.T 
    term4 = np.diag(term3)
    
    return np.mean(term2-term4)


def minMulDev(X, Y):
    n, d = X.shape
    k = Y.shape[1]
    W = np.zeros(d * k)
    res = optimize.minimize(mul_diviance, W, args=(X, Y, d, k))
    return np.reshape(res.x, (d, k))

#b
def classify(Xtest, W):
    scores = Xtest @ W
    max_indices = np.argmax(scores, axis=1)

    Yhat = np.zeros_like(scores)
    for index, max_index in enumerate(max_indices):
        Yhat[index][max_index] = 1
    return Yhat

#c
def calculateAcc(Yhat, Y):
    predicted = np.argmax(Yhat, axis=1)
    ground = np.argmax(Y, axis=1)
    return np.mean(predicted == ground)

'''probably remove when submitting'''
def synRegExperiment():
    return A3helpers.synClsExperiments(minMulDev, classify, calculateAcc)
    
# print(synRegExperiment())