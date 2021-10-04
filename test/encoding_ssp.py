import numpy as np
import sys
sys.path.append('..')

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils.data import fetch_data_with_label_per_step, label_to_int, balance_data
from utils.plotting import plot
from utils.clustering import agglomerative, kmeans, gaussian_mixture, Clustering
from utils.encoding import make_good_unitary, encode_feature, encode_dataset
from sklearn.metrics import adjusted_rand_score
import itertools
import argparse
import glob

def process_data(selected_data, selected_label, with_anchor, binding, aggregate, dim, wrap_feature = False, task = 'assembly'):
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

    assert dim is not None, 'Dimension of semantic pointer is not provided'
    assert binding is not None, 'Binding types for each entry is not provided'
    assert aggregate is not None, 'Aggregation types between features are not provided'
    x_axis_sp = make_good_unitary(dim)
    y_axis_sp = make_good_unitary(dim)
    z_axis_sp = make_good_unitary(dim)

    bolt_sp = make_good_unitary(dim)
    nut_sp = make_good_unitary(dim)

    table_sp = make_good_unitary(dim)
    jig_sp = make_good_unitary(dim)
    bin_origin_sp = make_good_unitary(dim)
    bin_target_sp = make_good_unitary(dim)
    if task == 'assembly':
        # Anchor object information
        ind_table = np.concatenate((np.where(selected_label_balanced == 'Nut on table')[0], np.where(selected_label_balanced == 'Bolt on table')[0]))
        ind_jig = np.delete(np.arange(n_data), ind_table)
        one_hot_anchor = np.zeros((n_data,2))
        one_hot_anchor[ind_table,0] = 1
        one_hot_anchor[ind_jig,1] = 1
        noise_anchor = np.random.normal(u_anchor, sigma_anchor, obj_kind.size)
        anchor_noisy = abs(one_hot_anchor - np.column_stack((noise_anchor, noise_anchor)))

        coord_sp = encode_feature(coord_noisy, [x_axis_sp, y_axis_sp, z_axis_sp], binding = binding[0], aggregate = aggregate[0])
        kind_sp = encode_feature(kind_noisy, [nut_sp, bolt_sp], binding = binding[1], aggregate = aggregate[1])
        anchor_sp = encode_feature(anchor_noisy, [table_sp, jig_sp], binding = binding[2], aggregate = aggregate[2])

    elif task == 'bin_picking':
        ind_bin_origin = np.where(selected_label_balanced == 'Object in origin bin')[0]
        ind_bin_target = np.delete(np.arange(n_data), ind_bin_origin)
        one_hot_anchor = np.zeros((n_data,2))
        one_hot_anchor[ind_bin_origin,0] = 1
        one_hot_anchor[ind_bin_target,1] = 1
        noise_anchor = np.random.normal(u_anchor, sigma_anchor, obj_kind.size)
        anchor_noisy = abs(one_hot_anchor - np.column_stack((noise_anchor, noise_anchor)))
        coord_sp = encode_feature(coord_noisy, [x_axis_sp, y_axis_sp, z_axis_sp], binding = binding[0], aggregate = aggregate[0])
        kind_sp = encode_feature(kind_noisy, [nut_sp, bolt_sp], binding = binding[1], aggregate = aggregate[1])
        anchor_sp = encode_feature(anchor_noisy, [bin_origin_sp, bin_target_sp], binding = binding[2], aggregate = aggregate[2])
    tag1 = make_good_unitary(dim)
    tag2 = make_good_unitary(dim)
    tag3 = make_good_unitary(dim)

    if wrap_feature == True:
        coord_sp = (tag1 * spa.SemanticPointer(coord_sp)).v
        kind_sp = (tag2 * spa.SemanticPointer(kind_sp)).v
        anchor_sp = (tag3 * spa.SemanticPointer(anchor_sp)).v
    if with_anchor:
        data_concat = [coord_sp, kind_sp, anchor_sp]
    else:
        data_concat = [coord_sp, kind_sp]
    return data_concat, selected_label_int

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
        data_concat, selected_label_int = process_data(selected_data, selected_label, with_anchor, binding, aggregate, dim, wrap_feature,task)
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
