import numpy as np
import sys
sys.path.append('..')

from utils.data import fetch_data_with_label_per_step, label_to_int, balance_data
from utils.clustering import Clustering
from utils.encoding import make_good_unitary, encode_feature
from sklearn.metrics import adjusted_rand_score
import itertools
import argparse
import glob

'''
run this file : python ./encoding_3d.py -t assembly -al gaussian -an True -af p_norm, euclidean, euclidean
'''

def process_data(selected_data, selected_label, with_anchor, task = 'assembly'):
    '''
    This method process the 4-d data(3-d coordinates + 1-d object kind) by:
     1. Balancing the data
     2. Add anchor object information if needed
     3. Change the index encoding(0 for nut, 1 for bolt) to distributed encodeing(ex. [0.8, 0.2] for nut and [0.9, 0.1] for bolt)
    '''
    u_coord = 0  # The average shift between the approximated coordinates and ground truth
    sigma_coord = 0.006
    u_kind = 0.1
    sigma_kind = 0.05
    u_anchor = 0.1
    sigma_anchor= 0.05

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
    if task == 'assembly':
        # Anchor object information
        ind_table = np.concatenate((np.where(selected_label_balanced == 'Nut on table')[0], np.where(selected_label_balanced == 'Bolt on table')[0]))
        ind_jig = np.delete(np.arange(n_data), ind_table)
        one_hot_anchor = np.zeros((n_data,2))
        one_hot_anchor[ind_table,0] = 1
        one_hot_anchor[ind_jig,1] = 1
        noise_anchor = np.random.normal(u_anchor, sigma_anchor, obj_kind.size)
        anchor_noisy = abs(one_hot_anchor - np.column_stack((noise_anchor, noise_anchor)))

    elif task == 'bin_picking':
        ind_bin_origin = np.where(selected_label_balanced == 'Object in origin bin')[0]
        ind_bin_target = np.delete(np.arange(n_data), ind_bin_origin)
        one_hot_anchor = np.zeros((n_data,2))
        one_hot_anchor[ind_bin_origin,0] = 1
        one_hot_anchor[ind_bin_target,1] = 1
        noise_anchor = np.random.normal(u_anchor, sigma_anchor, obj_kind.size)
        anchor_noisy = abs(one_hot_anchor - np.column_stack((noise_anchor, noise_anchor)))
    if with_anchor:
        data_concat = [coord_noisy, kind_noisy, anchor_noisy]
    else:
        data_concat = [coord_noisy, kind_noisy]
    return data_concat, selected_label_int

def test_3d(task, algorithm, with_anchor, affinity):
    if task == 'assembly':
        n_global_states = 4
        n_clusters = 5
        target_dir = '../../roboSim/data/assembly/*/*.pickle'
    elif task == 'bin_picking':
        n_global_states = 2
        n_clusters = 2
        target_dir = '../../roboSim/data/bin_picking/*/*.pickle'
    logfile_path = glob.glob(target_dir)
    #coordinates and object kind for nut and bolt
    data, label = fetch_data_with_label_per_step(logfile_path, n_global_states)
    n_test = 100
    result = []
    rand_score = 0
    n_success = 0
    n_skip = 0
    for i in range(n_test):
        ind = np.random.choice(len(data))
        selected_data = np.array(list(itertools.chain.from_iterable(data[ind])))
        selected_label = np.array(list(itertools.chain.from_iterable(label[ind])))
        if selected_data.shape[0] < 2:
            # Skip cases when there is only one datapoint
            n_skip += 1
            continue
        data_concat, selected_label_int = process_data(selected_data, selected_label, with_anchor, task)
        if with_anchor:
            affinity = affinity
        else:
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
            The success rate is {success_rate}% \n\
            The number of test is {n_test - n_skip}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = ' Test a specific case')
    parser.add_argument('-t','--task', type = str, help = 'The task to test. Options: assembly, bin_picking')
    parser.add_argument('-al','--algorithm', type = str, help = 'The algorithm to test. Options: agg, kmeans, gaussian')
    parser.add_argument('-an', '--with_anchor', type = str, help = 'Wheter or not including anchor object information. Options: True or False')
    parser.add_argument('-af', '--affinity', nargs='+', help = 'The affinity to test. Options: p_norm, euclidean, cosine')

    args = parser.parse_args()
    if len(args.affinity) == 1: # when one affinity is used for all features
        args.affinity = 3 * args.affinity
    test_3d(args.task, args.algorithm, args.with_anchor, args.affinity)
