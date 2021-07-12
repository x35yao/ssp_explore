import numpy as np
from sklearn.metrics import pairwise_distances

def p_norm_distance(X, Y = None, p = -3):
    n = X.shape[0]
    if Y is None:
        distances = np.zeros((n,n))
        for i in range(n):
            for j in range(i+1,n):
                difference = np.abs(X[i, :] - X[j, :])
                distances[i,j] = np.sum(difference ** p) ** (1 / p)
        distances = distances + distances.T
    else:
        distances = np.zeros(n)
        for i in range(n):
            difference = np.abs(X[i, :] - Y)
            distances[i] = np.sum(difference ** p) ** (1 / p)
    return distances

def cos_distance(X, Y = None):
    return pairwise_distances(X, Y, metric = 'cosine')

def euclidean_distance(X, Y = None):
    return pairwise_distances(X, Y, metric = 'euclidean')

def pairwise_distance(X, Y = None, affinity = 'euclidean', p = -3):
    if affinity == 'p_norm':
        return p_norm_distance(X, Y, p)
    elif affinity == 'cosine':
        return cos_distance(X, Y)
    elif affinity == 'euclidean':
        return euclidean_distance(X, Y)
