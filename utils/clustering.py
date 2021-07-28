from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import distance_metrics
import numpy as np
from .distances import p_norm_distance, cos_distance, euclidean_distance, pairwise_distance

def agglomerative(X, affinity, thres, n_clusters = None, p = -3):
    '''
    X: A list of n data features. Each entry is An array of data of size n_samples * n_dimension. Different affinity is applied for differnt feature.
    thres: The distance thresholed to seperate two clusters.
    affinity: A list of n affinities. The distance metric to seperate clusters. Could be 'p_norm', 'euclidean', 'cosine'.
    n_clusters: The result number of clusters.
    p: the p value if p-norm is used as affinity

    return: The extimated labels for each datapoint.
    '''
    if len(set(affinity)) == 1: # Apply the same affinity to different features.
        affinity = affinity[0]
        if isinstance(X, list):
            X = np.concatenate(X, axis = 1)
        if affinity in distance_metrics():
            ac = AgglomerativeClustering(n_clusters = n_clusters, affinity= affinity, linkage='average', distance_threshold= thres)
            estimated_labels = ac.fit_predict(X)
        elif affinity == 'p_norm':
            ac = AgglomerativeClustering(n_clusters = n_clusters, affinity= 'precomputed', linkage='average', distance_threshold= thres)
            distances = p_norm_distance(X, p = p)
            estimated_labels = ac.fit_predict(distances)
    elif len(set(affinity)) != 1: # Apply different affinities to different features.
        ac = AgglomerativeClustering(n_clusters = n_clusters, affinity= 'precomputed', linkage='average', distance_threshold= thres)
        n_data = X[0].shape[0]
        distances = np.zeros((n_data, n_data))
        for i, data in enumerate(X):
            if affinity[i] == 'p_norm':
                distances += p_norm_distance(X[i], p = p)
            elif affinity[i] == 'euclidean':
                distances += euclidean_distance(X[i])
            elif affinity[i] == 'cosine':
                distances += cos_distance(X[i])
        estimated_labels = ac.fit_predict(distances)
    return estimated_labels

def get_dist_to_centers(X, centers, affinity):
    '''
    X: An array of data of size n_samples * n_dimension
    Centers: An array with size n_samples * n_clusters

    Return: dist_to_centers of size n_samples * n_clusters
    '''

    n_samples = X.shape[0]
    n_clusters = centers.shape[0]
    dist_to_centers = np.zeros((n_samples, n_clusters))
    for i in range(n_clusters):
        center = centers[i,:].reshape(1, -1)
        dist_to_centers[:, i] = pairwise_distance(X, center, affinity).flatten()
    return dist_to_centers

def get_new_centers(X, dist_to_centers):
    d = X.shape[1]
    n_clusters = dist_to_centers.shape[1]
    labels = np.argmin(dist_to_centers, axis = 1)
    centers_new = np.zeros((n_clusters, d))
    for i in range(n_clusters):
        cluster_ind = np.where(labels == i)[0]
        n_members = cluster_ind.shape[0]
        if n_members == 0: # Handle empty cluster by randomly pick a center in the dataset.
            centers_new[i] = X[np.random.choice(X.shape[0])]
        else:
            cluster_members = X[cluster_ind,:] # with shape n_members * n_feature_dimension
            centers_new[i] = np.mean(cluster_members, axis = 0)
    return centers_new

def kmeans(X, n_clusters, affinity, init_centroids = 'random', n_init = 10, max_iter = 300, tol = 0.0001, p = -3):
    '''
    X: An array of data of size n_samples * n_dimension
    n_clusters: The result number of clusters.
    affinity: the metric used to metrue distance between datapints. 'p_norm', 'euclidean', 'cosine'
    init_centoids: The indicies of datapoints that are initialized as the centroids.
    n_init: Number of time the k-means algorithm will be run with different centroid seeds.
    max_iter: Maximum number of iterations of the k-means algorithm for a single run.
    tol: Relative tolerance with regards to Frobenius norm of the difference in the cluster centers of two consecutive iterations to declare convergence.
    p: the p value if p-norm is used as affinity

    return:
    labels :The extimated labels for each datapoint.
    centers: The centers of each cluster.
    '''
    if len(set(affinity)) == 1: # Apply the same affinity to different features.
        affinity = affinity[0]
        if isinstance(X, list):
            X = np.concatenate(X, axis = 1)
        n_samples = X.shape[0]
        result = {}
        for i in range(n_init):
            diff = 1
            itr = 0
            # initize centoids
            if init_centroids == 'random':
                ind = np.random.choice(n_samples, n_clusters, replace = False)
            else:
                ind = init_centroids
            centers = X[ind, :]
            while diff > tol and itr < max_iter:
                dist_to_centers = get_dist_to_centers(X, centers, affinity)
                labels = np.argmin(dist_to_centers, axis = 1) # assign datapoint to clusters.
                centers_new = get_new_centers(X, dist_to_centers)
                center_diff = ((centers_new - centers)**2).sum()
                centers = centers_new
                itr +=1
            dist_to_centers = get_dist_to_centers(X, centers, affinity)
            labels = np.argmin(dist_to_centers, axis = 1)
            dist_tol = np.min(dist_to_centers, axis = 1).sum() # total distances of all points to their assigned center
            result[dist_tol]= labels
        opt_ind = min(result.keys())
    elif len(set(affinity)) != 1: # Apply different affinities to different features.
        n_samples = X[0].shape[0]
        result = {}
        for i in range(n_init):
            diff = 1
            itr = 0
            # initize centoids
            if init_centroids == 'random':
                ind = np.random.choice(n_samples, n_clusters, replace = False)
            else:
                ind = init_centroids
            centers = []
            for feature in X:
                centers.append(feature[ind,:])
            while diff > tol and itr < max_iter:
                dist_to_centers = []
                for i, feature in enumerate(X):
                    dist_to_centers.append(get_dist_to_centers(X[i], centers[i], affinity[i]))
                labels = np.argmin(sum(dist_to_centers), axis = 1) # assign datapoint to clusters.
                centers_new = []
                for i, feature in enumerate(X):
                    centers_new.append(get_new_centers(X[i], dist_to_centers[i]))
                center_diff = sum([((centers_new[i] - centers[i])**2).sum() for i, center in enumerate(X)])
                centers = centers_new
                itr +=1
            dist_to_centers = []
            for i, feature in enumerate(X):
                dist_to_centers.append(get_dist_to_centers(X[i], centers[i], affinity[i]))
            labels = np.argmin(sum(dist_to_centers), axis = 1)
            dist_tol = np.min(sum(dist_to_centers), axis = 1).sum() # total distances of all points to their assigned center
            result[dist_tol]= labels
        opt_ind = min(result.keys())

    return result[opt_ind], centers

def gaussian_mixture(X, n_clusters):
    '''
    This function cluster the data X using gaussian mixture model.

    X: the input data
    n_clusters: number of clusters of the data

    return:
    estimated_label: the estimated label for each data
    em: the em object which has attribute means, covariances, etc(see sklearn.mixture.GaussianMixture).
    '''
    if isinstance(X, list):
        X = np.concatenate(X, axis = 1)
    em = GaussianMixture(n_clusters)
    estimated_label = em.fit_predict(X)
    return estimated_label, em

class Clustering:
    def __init__(self, X, algorithm, affinity, n_clusters, thres, p = -2):
        self.data = X
        self.algorithm = algorithm
        self.n_clusters = n_clusters
        self.thres = thres
        self.affinity = affinity
        self.p = p
    def predict(self):
        if self.algorithm == 'agg':
            self.estimated_labels = agglomerative(self.data, self.affinity, self.thres, self.n_clusters, self.p)
        elif self.algorithm == 'kmeans':
            self.estimated_labels, self.centers = kmeans(self.data, self.n_clusters, self.affinity, init_centroids = 'random', n_init = 10, max_iter = 300, tol = 0.0001, p = self.p)
        elif self.algorithm == 'gaussian':
            (self.estimated_labels,_) = gaussian_mixture(self.data, self.n_clusters)
        return self.estimated_labels
