import numpy as np
import cvxopt.solvers as solvers
import scipy.optimize as optimize
import scipy.linalg as linalg
import scipy.special._logsumexp as log
import pandas as pd
import A3helpers
import scipy.spatial.distance as dist

solvers.options['show_progress'] = False

#1
#a
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

#2
#a
def PCA(X, k):
    n, d = X.shape
    mu = np.mean(X, axis=0)
    U = X - mu
    res = linalg.eigh(U.T @ U)
    v = res[1].T
    return v[::-1][:k]

#b
def fashion_MNIST():
    dataset = pd.read_csv("A3train.csv").to_numpy()
    U = PCA(dataset, 20)
    A3helpers.plotImgs(U)

#c 
def projPCA(Xtest, mu, U):
    Xproj = (Xtest - mu.T)@U.T
    return Xproj

#d
def synClsExperimentsPCA():
    n_runs = 100
    n_train = 128
    n_test = 1000
    dim_list = [1, 2]
    gen_model_list = [1, 2]
    train_acc = np.zeros([len(dim_list), len(gen_model_list), n_runs])
    test_acc = np.zeros([len(dim_list), len(gen_model_list), n_runs])
    for r in range(n_runs):
        for i, k in enumerate(dim_list):
            for j, gen_model in enumerate(gen_model_list):
                Xtrain, Ytrain = A3helpers.generateData(n=n_train, gen_model=gen_model)
                Xtest, Ytest = A3helpers.generateData(n=n_test, gen_model=gen_model)
                U = PCA(Xtrain, k)
                mu = np.mean(Xtrain, axis=0)
                Xtrain_proj = projPCA(Xtrain, mu, U) # TODO: call your projPCA to find the new features
                Xtest_proj = projPCA(Xtest, mu, U) # TODO: call your projPCA to find the new features
                Xtrain_proj = A3helpers.augmentX(Xtrain_proj) # add augmentation
                Xtest_proj = A3helpers.augmentX(Xtest_proj)
                W = minMulDev(Xtrain_proj, Ytrain) # from Q1
                Yhat = classify(Xtrain_proj, W) # from Q1
                train_acc[i, j, r] = calculateAcc(Yhat, Ytrain) # from Q1
                Yhat = classify(Xtest_proj, W)
                test_acc[i, j, r] = calculateAcc(Yhat, Ytest)
    
    train_accuracy = np.mean(train_acc, axis=2)
    test_accuracy = np.mean(test_acc, axis=2)

    return train_accuracy, test_accuracy
    # TODO: compute the average accuracies over runs
    # TODO: return 2-by-2 train accuracy and 2-by-2 test accuracy

#3
#a
def kmeans(X, k, max_iter=1000):
    n, d = X.shape
    assert max_iter > 0

    indices= np.random.choice(n, k, replace=False)  
    U = X[indices]                                       # TODO: Choose k random points from X as initial centers
    for i in range(max_iter):
        D = dist.cdist(X, U)            # TODO: Compute pairwise distance between X and U
        Y = cluster_assignments(D)              # TODO: Find the new cluster assignments
        old_U = U
        U = np.linalg.inv((Y.T @ Y + 1e-8 * np.eye(k)))@ Y.T @ X   # TODO: Update cluster centers
        if np.allclose(old_U, U):
            break
    W = X - Y@U
    obj_val = 1/(2*n) * np.linalg.norm(W, 'fro') ** 2   # TODO: Compute objective value
    return Y, U, obj_val

#b
def cluster_assignments(D):
    max_indices = np.argmin(D, axis=1)

    Y = np.zeros_like(D)
    for index, max_index in enumerate(max_indices):
        Y[index][max_index] = 1
    return Y


#b
def repeatKmeans(X, k, n_runs=100):
    best_obj_val = float('inf')
    best_y = None
    best_u = None
    for r in range(n_runs):
        Y, U, obj_val = kmeans(X, k)
        # TODO: Compare obj_val with best_obj_val. If it is lower,
        # then record the current Y, U and update best_obj_val
        if obj_val < best_obj_val:
            best_obj_val = obj_val
            best_y = Y
            best_u = U
    return best_y, best_u, best_obj_val
    # TODO: Return the best Y, U and best_obj_val

#c
def chooseK(X, k_candidates=[2,3,4,5,6,7,8,9]):
    obj_val_list = []
    for k in k_candidates:
        _, _, obj_val = repeatKmeans(X,k)
        obj_val_list.append(obj_val)

    return obj_val_list