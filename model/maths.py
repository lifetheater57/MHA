import numpy as np

def relu(X):
    X[X < 0] = 0
    return X

def project_positive(X):
    max_idx = X.argmax(axis=1)
    for i in range(X.shape[0]):
        X[i, X[i,:] < X[i, max_idx[i]]] = 0
        X = relu(X)
    return X

def normalize_cols(X, eps=0.001):
    X = X / (np.linalg.norm(X, axis=0) + eps)
    return X