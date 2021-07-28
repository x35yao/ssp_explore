import numpy as np
import sys
sys.path.append('..')

from utils.data import fetch_data_with_label_per_step, label_to_int, balance_data
from utils.clustering import Clustering
from utils.encoding import make_good_unitary, encode_feature
from sklearn.metrics import adjusted_rand_score
import itertools
import argparse

def test_3d(algorithm, with_anchor, affinity):
    '''
    run this: python .\3d_encoding.py -al gaussian -an True -af p_norm, euclidean, euclidean
    '''
    logfile_path = ['../data/raw/1599166289/data.pickle', '../data/raw/1599153598/data.pickle', '../data/raw/test/data.pickle']

    #coordinates and object kind for nut and bolt
    data, label = fetch_data_with_label_per_step(logfile_path, force_success = True)
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
        if with_anchor:
            data_concat = [coord_noisy, kind_noisy, anchor_noisy]
            affinity = affinity
        else:
            data_concat = [coord_noisy, kind_noisy]
            affinity = affinity[:2]

        C = Clustering(data_concat, algorithm, affinity, n_clusters, thres = None, p = -2)
        estimated_label = C.predict()

        n_original = selected_data.shape[0]
        temp = adjusted_rand_score(estimated_label[:n_original], selected_label_int[:n_original])
        rand_score += temp
        if temp == 1:
            n_success +=1
        average_rand = rand_score / n_test
        success_rate = n_success / n_test * 100
    print(f'Algorithm is {algorithm}, Affinity is {affinity}, With anchor is {with_anchor}. \n \
            The average rand score is {average_rand}.\n \
            The success rate is {success_rate}%')
    return average_rand, success_rate
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = ' Test a specific case')
    parser.add_argument('-al','--algorithm', type = str, help = 'The algorithm to test. Options: agg, kmeans, gaussian')
    parser.add_argument('-an', '--with_anchor', type = str, help = 'Wheter or not including anchor object information. Options: True or False')
    parser.add_argument('-af', '--affinity', nargs='+', help = 'The affinity to test. Options: p_norm, euclidean, cosine')

    args = parser.parse_args()
    if len(args.affinity) == 1: # when one affinity is used for all features
        args.affinity = 3 * args.affinity
    test_3d(args.algorithm, args.with_anchor, args.affinity)
