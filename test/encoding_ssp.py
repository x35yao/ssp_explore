import numpy as np
import sys
sys.path.append('..')

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.data import fetch_data_with_label, label_to_int, balance_data , process_data_ssp
from utils.plotting import plot
from utils.clustering import agglomerative, kmeans, gaussian_mixture, Clustering
from utils.encoding import make_good_unitary, encode_feature, encode_dataset
from sklearn.metrics import adjusted_rand_score
import itertools
import argparse
import glob

'''
run this code: python ./encoding_ssp.py -t bin_picking -al gaussian -an False -af cosine -b power multiply multiply -ag multiply sum sum -ag_bf sum -d 256 -w True
'''

def test_ssp(task, algorithm, with_anchor, affinity, binding, aggregate, aggregate_between_feature, dim, wrap_feature):
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
    data, label = fetch_data_with_label(logfile_path, n_global_states)
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
        data_concat, selected_label_int = process_data_ssp(selected_data, selected_label, with_anchor, binding, aggregate, dim, wrap_feature,task)
        data_encoded = encode_dataset(data_concat, aggregate_between_feature)
        C = Clustering(data_encoded, algorithm, affinity, n_clusters, thres = None, p = -2)
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
    parser.add_argument('-al', '--algorithm', type = str, help = 'The algorithm to test. Options: agg, kmeans, gaussian')
    parser.add_argument('-an','--with_anchor', type = str, help = 'Wheter or not including anchor object information. Options: True or False')
    parser.add_argument('-af','--affinity', nargs='+', help = 'The affinity to test. Options: p_norm, euclidean, cosine')
    parser.add_argument('-b','--binding', nargs='+', help = 'The way to bind each entry. Options: multiply, power')
    parser.add_argument('-ag','--aggregate', nargs = '+', help = 'The way to bind between entries. Options: sum, multiply')
    parser.add_argument('-ag_bf','--aggregate_between_feature', type = str, help = 'The way to bind between features. Options: sum, convolution')
    parser.add_argument('-d','--dim', type = int, help = 'The dimension of the semantic pointers.')
    parser.add_argument('-w','--wrap_feature', type = str, help = 'Wheter or not wrap each feature with a tag')
    args = parser.parse_args()

    assert args.dim is not None, 'Dimension of semantic pointer is not provided'
    assert args.algorithm in ['agg', 'kmeans', 'gaussian'], 'algorithm invalided'
    if len(args.binding) == 1:
        args.binding = args.binding * 3
    if len(args.aggregate) == 1:
        args.aggregate = args.aggregate * 3
    test_ssp(args.task, args.algorithm, args.with_anchor, args.affinity, args.binding, args.aggregate, args.aggregate_between_feature, args.dim, args.wrap_feature)
