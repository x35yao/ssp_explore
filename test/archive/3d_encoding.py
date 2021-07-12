import numpy as np
import sys
sys.path.append('..')

from utils.data import fetch_data_with_label_per_step, label_to_int, balance_data
from utils.clustering import agglomerative, kmeans, gaussian_mixture
from utils.encoding import make_good_unitary, encode_feature
from sklearn.metrics import adjusted_rand_score
import itertools
import argparse

logfile_path = ['../data/raw/1599166289/data.pickle', '../data/raw/1599153598/data.pickle', '../data/raw/test/data.pickle']

#coordinates and object kind for nut and bolt
data, label = fetch_data_with_label_per_step(logfile_path)

parser = argparse.ArgumentParser(description = ' Test a specific case')
parser.add_argument('algorithm', type = str, help = 'The algorithm to test. Options: agg, kmeans, gaussian')
parser.add_argument('with_anchor', type = str, help = 'Wheter or not including anchor object information. Options: True or False')
parser.add_argument('-affinity', type = str, help = 'The affinity to test. Options: p_norm, euclidean, cosine')

args = parser.parse_args()

n_clusters = 5
n_test = 100
result = []
u_coord = 0  # The average shift between the approximated coordinates and ground truth
sigma_coord = 0.006
u_kind = 0.1
sigma_kind = 0.05
u_anchor = 0.1
sigma_anchor= 0.05

rand_score = 0
n_success = 0
for i in range(n_test):
    ind = np.random.choice(len(data))
    selected_data = np.array(list(itertools.chain.from_iterable(data[ind])))
    selected_label = np.array(list(itertools.chain.from_iterable(label[ind])))
    selected_data_balanced, selected_label_balanced = balance_data(selected_data, selected_label)
    selected_label_int = label_to_int(selected_label_balanced)

    # Object coordinates information
    coord = selected_data_balanced[:,0:3]
    obj_kind = selected_data_balanced[:, 3]
    noise = np.random.normal(u_coord, sigma_coord, coord.shape)
    coord_noisy = coord + noise

    # Object kind information
    ind_nut = np.where(obj_kind == 0)[0]
    ind_bolt = np.where(obj_kind == 1)[0]
    n_data = coord.shape[0]
    one_hot_kind = np.zeros((n_data,2))
    one_hot_kind[np.arange(obj_kind.size),obj_kind.astype(int)] = 1
    noise_kind = np.random.normal(u_kind, sigma_kind, obj_kind.size)
    kind_noisy = abs(one_hot_kind - np.column_stack((noise_kind, noise_kind)))

    # Anchor object information
    ind_table = np.concatenate((np.where(selected_label_balanced == 'Nut on table')[0], np.where(selected_label_balanced == 'Bolt on table')[0]))
    ind_jig = np.delete(np.arange(n_data), ind_table)
    one_hot_anchor = np.zeros((n_data,2))
    one_hot_anchor[ind_table,0] = 1
    one_hot_anchor[ind_jig,1] = 1
    noise_anchor = np.random.normal(u_anchor, sigma_anchor, obj_kind.size)
    anchor_noisy = abs(one_hot_anchor - np.column_stack((noise_anchor, noise_anchor)))
    if args.with_anchor:
        data_concat = np.concatenate((coord_noisy, kind_noisy, anchor_noisy), axis = 1)
    else:
        data_concat = np.concatenate((coord_noisy, kind_noisy), axis = 1)

    if args.affinity == 'compound':
        '''TODO: the case where differet features use different affinity'''
        pass
    else:  # One affinity is used for differnet features.
        if args.algorithm == 'agg':
            assert args.affinity is not None, 'Affinity is not provided.'
            p = -3
            estimated_label = agglomerative(data_concat, affinity = args.affinity, thres = None, n_clusters = n_clusters, p = p)
        elif args.algorithm  == 'kmeans':
            assert args.affinity is not None, 'Affinity is not provided.'
            estimated_label, centers= kmeans(data_concat, n_clusters = n_clusters , affinity = args.affinity)
        elif args.algorithm == 'gaussian':
            estimated_label, centers = gaussian_mixture(data_concat, n_clusters = n_clusters)

    n_original = selected_data.shape[0]
    temp = adjusted_rand_score(estimated_label[:n_original], selected_label_int[:n_original])
    rand_score += temp
    if temp == 1:
        n_success +=1
print(f'Algorithm is {args.algorithm}, Affinity is {args.affinity}, With anchor is {args.with_anchor}. \n \
        The average rand score is {rand_score / n_test}.\n \
        The success rate is {n_success / n_test * 100}%')
