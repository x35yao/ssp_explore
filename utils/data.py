import numpy as np
import pickle
from scipy.stats import trim_mean
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import MultiLabelBinarizer
import itertools
from utils.encoding import make_good_unitary, encode_feature, encode_dataset

def fetch_data(x, n_global_states = None):
    '''
    This function returns the coordinates of bolts and nuts with NO labels.

    x: a list of file_path
    n_global_states: If n_global_states is given, it will only take the data from successful attempt(n_attempt = n_global_states)
               which will then reduce the number of data but also balance the data
    return: coord_nut, coord_bolt
    '''
    seqs = []
    for file_path in x:
        with open(file_path, "rb") as logfile:
            while True:
                try:
                    seqs.append(pickle.load(logfile))
                except EOFError:
                    break
    seqs_success = []
    if n_global_states != None:
        for seq in seqs:
            if len(seq) == n_global_states:
                seqs_success.append(seq)
        seqs = seqs_success
    data_nut = []
    data_bolt = []
    for seq in seqs:
        for step in seq:
            temp_nut = [list(obj['pos']) + list(obj['orn']) for obj in step['objs'] if obj['class'] == 'nut']
            temp_bolt = [list(obj['pos']) + list(obj['orn']) for obj in step['objs'] if obj['class'] == 'bolt']
            data_nut += temp_nut
            data_bolt += temp_bolt

    data_nut = np.asarray(data_nut)
    data_bolt = np.asarray(data_bolt)

    coord_nut = data_nut[:,:3]
    coord_bolt = data_bolt[:,:3]
    delta = 0.01
    for i, val in enumerate(coord_bolt[:,1]):
        if val > 0.064:
            coord_bolt[i][1] += delta

    return coord_nut, coord_bolt

# def fetch_data_with_label(x, n_global_states = None):
#     '''
#     x: a list of file_path
#     n_global_states: If n_global_states is given, it will only take the data from successful attempt(n_attempt = n_global_states)
#                which will then reduce the number of data
#     return: x_nut, y_nut, x_bolt, y_bolt
#     '''
#     seqs = []
#     for file_path in x:
#         with open(file_path, "rb") as logfile:
#             while True:
#                 try:
#                     seqs.append(pickle.load(logfile))
#                 except EOFError:
#                     break
#     seqs_success = []
#     if n_global_states != None:
#         for seq in seqs:
#             if len(seq) == n_global_states:
#                 seqs_success.append(seq)
#         seqs = seqs_success
#     test_nut = []
#     test_bolt = []
#     for seq in seqs:
#         for step in seq:
#             if step['action'] == 'PUT_NUT_IN_JIG':
#                 data = [obj['pos'] for obj in step['objs'] if obj['class'] == 'nut'][0]
#                 test_nut.append(list(data) + ['Nut on table'])
#             elif step['action'] == 'PUT_BOLT_IN_JIG':
#                 data = [obj['pos'] for obj in step['objs'] if obj['class'] == 'bolt'][0]
#                 if data[1] < 0.03:
#                     test_bolt.append(list(data) + ['Bolt on table'])
#             elif step['action'] == 'ASSEMBLE':
#                 data_nut = [obj['pos'] for obj in step['objs'] if obj['class'] == 'nut'][0]
#                 if data_nut[1] < 0.02: # Wrong labeld data
#                     pass
#                 else:
#                     test_nut.append(list(data_nut) + ['Nut in jig'])
#                 data_bolt = [obj['pos'] for obj in step['objs'] if obj['class'] == 'bolt'][0]
#                 if data_bolt[1] < 0.06: # Wrong labeld data
#                     pass
#                 else:
#                     test_bolt.append(list(data_bolt) + ['Bolt in jig'])
#             elif step['action'] == 'ASSEMBLED' or step['action'] == None:
#                 data = [obj['pos'] for obj in step['objs'] if obj['class'] == 'bolt'][0]
#                 # test_bolt.append(list(data) + ['Bolt assembled'])
#                 if data[2] > -0.6:
#                     pass
#                 else:
#                     test_bolt.append(list(data) + ['Bolt assembled'])
#     data_nut = np.array(test_nut)
#     data_bolt = np.array(test_bolt)
#     x_nut = data_nut[:,:3].astype(np.float)
#     y_nut = data_nut[:,3]
#     x_bolt = data_bolt[:,:3].astype(np.float)
#     y_bolt = data_bolt[:,3]
#     # Seperate data a little more
#     delta = 0.01
#     for i, val in enumerate(x_bolt[:,1]):
#         if val > 0.064:
#             x_bolt[i][1] += delta
#     return x_nut, y_nut, x_bolt, y_bolt

def fetch_data_with_label(x, n_global_states = None):
    '''
    x: a list of file_path
    n_global_states: If n_global_states is given, it will only take the data from successful attempt(n_attempt = n_global_states)
               which will then reduce the number of data but also balance the data

    return: data_concat: a list with lengh t,  where t is the number of trials. Each entry is is a list with len s, where s(is 4 here) is the
                         number of steps per trial and each entry of this list is another list with lentgh n, where n is the number of items. Each
                         entry is a list with length d, where d is the dimension of the data.
                         d = 4 for now. The first 3 entries are the 3D coordinates: x, z, y. The 4th. entry is indicating wheter
                         the item is a bolt(1) or nut(0).

            label: a list with lengh t, each entry is an s * n array that indicates the label of each object(bolt/ nut).
                   The label could be 'Bolt on table' etc.
    '''
    seqs = []
    for file_path in x:
        with open(file_path, "rb") as logfile:
            while True:
                try:
                    seqs.append(pickle.load(logfile))
                except EOFError:
                    break
    seqs_success = []
    if n_global_states != None:
        for seq in seqs:
            if len(seq) == n_global_states:
                seqs_success.append(seq)
        seqs = seqs_success
    data_concat = []
    label = []
    for j, seq in enumerate(seqs):
        data_concat_seq = []
        label_seq = []
        for step in seq:
            label_nut = []
            label_bolt = []
            data_nut = [list(obj['pos']) + [0] for obj in step['objs'] if obj['class'] == 'nut']
            data_bolt = [list(obj['pos']) + [1] for obj in step['objs'] if obj['class'] == 'bolt']
            if step['action'] == 'PUT_NUT_IN_JIG':
                for i in data_nut:
                    label_nut.append('Nut on table')
                for i in data_bolt:
                    label_bolt.append('Bolt on table')
            elif step['action'] == 'PUT_BOLT_IN_JIG':
                for i in data_nut:  # Skechy here, some trial starts from 'Put bolt in jig' and don't have 'Put nut in jig'. Might be troublesome.
                    label_nut.append('Nut on table')
                for i in data_bolt:
                    label_bolt.append('Bolt on table')
                if data_nut[0][1] > 0.04:
                    label_nut[0] = 'Nut in jig'
            elif step['action'] == 'ASSEMBLE':
                for i in data_nut:
                    label_nut.append('Nut on table')
                for i in data_bolt:
                    label_bolt.append('Bolt on table')
                if data_nut[0][1] > 0.04:
                    label_nut[0] = 'Nut in jig'
                if data_bolt[0][1] > 0.06:
                    label_bolt[0] = 'Bolt in jig'
            elif step['action'] == 'ASSEMBLED' or step['action'] == None:
                for i in data_nut:  # Skechy here, some trial starts from 'Put bolt in jig' and don't have 'Put nut in jig'. Might be troublesome.
                    label_nut.append('Nut on table')
                for i in data_bolt:
                    label_bolt.append('Bolt on table')
                if data_nut[0][1] > 0.04:
                    label_nut[0] = 'Nut in jig'
                if data_bolt[0][1] > 0.06:
                    label_bolt[0] = 'Bolt assembled'
            elif step['action'] == 'CLEAN_BOLT' or step['action'] == 'CLEAN_NUT' or step['action'] == 'Cleaned':
                pos_bin_origin = [list(obj['pos'])  for obj in step['objs'] if obj['class'] == 'bin_origin'][0]
                pos_bin_target = [list(obj['pos'])  for obj in step['objs'] if obj['class'] == 'bin_target'][0]
                r = 0.4

                ind_to_remove_nut = []
                ind_to_remove_bolt = []
                for i, pos_obj in enumerate(data_nut):
                    if is_in_bin(pos_obj, pos_bin_origin, r):
                        label_nut.append('Object in origin bin')
                    elif is_in_bin(pos_obj, pos_bin_target, r):
                        label_nut.append('Object in target bin')
                    else:
                        # delete noisy data
                        label_nut.append('Noise data')
                for j, pos_obj in enumerate(data_bolt):
                    if is_in_bin(pos_obj, pos_bin_origin, r):
                        label_bolt.append('Object in origin bin')
                    elif is_in_bin(pos_obj, pos_bin_target, r):
                        label_bolt.append('Object in target bin')
                    else:
                        label_bolt.append('Noise data')
            data_concat_seq.append(data_nut + data_bolt)
            label_seq.append(label_nut + label_bolt)
        # data_concat_seq = np.array(data_concat_seq)
        # label_seq = np.array(label_seq)

        data_concat.append(data_concat_seq)
        label.append(label_seq)

    return data_concat, label
def label_to_int(labels, label_seq = None):
    '''
    This function will convert an array of labels(which are strs) to an array of integers.
    label_seq: defines which label stands for which integer.
    '''

    if isinstance(label_seq, list):
        result = np.zeros((labels.shape[0]))
        for i, label in enumerate(label_seq):
            result[np.where(labels == label)] = i
        return result.astype(int)
    else:
        result = np.zeros((labels.shape[0]))
        for i, label in enumerate(set(labels)):
            result[np.where(labels == label)] = i
        return result.astype(int)

def int_to_label(labels, label_seq):
    '''
    This function will convert an array of labels(which are strs) to an array of integers.
    label_seq: defines which label stands for which integer.
    '''
    result = []
    for i, int_label in enumerate(labels):
        result.append(label_seq[int_label])
    return np.array(result)

# def divide_data(x, y, n_samples):
#     '''
#     Divide data into training set and test set.
#     x: all the data with shape n_data by n_dimension
#     y: lables for all the data with n_data entries
#     n_samples: number of samples for training set
#
#     return:
#     x_train, y_train, x_test, y_test
#     '''
#     n_data = x.shape[0]
#     n_training = len(set(y)) * n_samples
#     n_test = n_data - n_training
#     x_train = np.empty((n_training, 3))
#     y_train = np.empty(n_training, dtype = y.dtype)
#     x_test = np.empty((n_test,3))
#     y_test = np.empty(n_test, dtype = y.dtype)
#
#     train_inds = []
#     for i, label in enumerate(set(y)):
#         ind = np.where(y == label)[0]
#         selected_ind = np.random.choice(ind, n_samples, replace = False)
#         train_inds += selected_ind.tolist()
#         x_train[i * n_samples:(i + 1) * n_samples,:] = x[selected_ind, :]
#         y_train[i * n_samples:(i + 1) * n_samples] = y[selected_ind]
#     inds = np.arange(n_data)
#     test_inds = np.delete(inds, np.array(train_inds))
#     x_test = x[test_inds,:]
#     y_test = y[test_inds]
#
#     return x_train, y_train, x_test, y_test

def divide_data(x, y, n_samples):
    '''
    Divide data into training set and test set.
    x: list of data with length n. n is the number of trials recorded
    y: list of labels with length n. For each entry, labels at each step are recorded.
    n_samples: number of samples for training set

    return:
    x_train, y_train, x_test, y_test. They are all lists(maybe change to np array later?).
    '''
    n_data = len(x)
    ind_train = np.random.choice(n_data, n_samples)
    ind_test = np.delete(np.arange(n_data), ind_train)
    if isinstance(x, list):
        x_train = [x[i] for i in ind_train]
        y_train = [y[i] for i in ind_train]
        x_test = [x[i] for i in ind_test]
        y_test = [y[i] for i in ind_test]
    elif isinstance(x, np.ndarray):
         # x is an array
         x_train = x[ind_train, :]
         y_train = y[ind_train]
         x_test = x[ind_test, :]
         y_test = y[ind_test]

    return x_train, y_train, x_test, y_test

def build_model(estimated_labels, data_concat):
    '''
    Given estimated labels and the data. This function find the mean, covirance and label for each cluster.
    '''
    models = []
    # model the distribution of each cluster robustly in case there are misclassifications
    for label in np.unique(estimated_labels):
        indices = np.nonzero(estimated_labels == label)
        samples = data_concat[indices,:].squeeze()
        if samples.ndim == 1:
            means = samples
            covariance = np.zeros((3,3))
        else:
            means = trim_mean(samples, 0, axis=0)
            mcd = MinCovDet(support_fraction=1).fit(samples)
            covariance = mcd.covariance_
        models.append([means, covariance, label - min(estimated_labels)])
    return models

def generate_data(models, n_samples, variance_level):
    '''
    Given mean, covirance(saved in models), this function generate n_samples samples with variance_level virance.

    models: For each cluster, models[0] is the mean, models[1] is the covariance, models[2] is the label.
    n_samples: number of samples need to be generated.
    varance_level: the variance level when gererating new samples.

    return:
    x: generated samples
    y: labels of generated samples
    '''
    x = []
    y = []
    for model in models:
        x.append(np.random.multivariate_normal(model[0][:3] , model[1][:3,:3] * variance_coefficient , n_samples))
        y += [model[2]] * n_samples
    return x, y

def balance_data(data, labels):
    count = []
    for label in set(labels):
        count.append((labels == label).sum())
    max_count = max(count)
    for label in set(labels):
        n_data = (labels == label).sum()
        if n_data < max_count:
            ind = np.where(labels == label)[0]
            selected_data = data[ind, :]
            average = np.average(selected_data, axis = 0)
            n_needed = max_count - n_data
            data_needed = np.tile(average, (n_needed, 1))
            label_needed = np.tile(label, n_needed)
            data = np.concatenate((data, data_needed), axis = 0)
            labels = np.concatenate((labels, label_needed))
    return data, labels

def is_in_bin(pos_obj, pos_bin, r):
    # The object center of the bin is 0.15 off in the x direction
    offset = 0.15
    dist_to_bin = np.sqrt((pos_obj[0] - (pos_bin[0] - offset))**2 + (pos_obj[2] - pos_bin[2])**2)
    # print(obj,pos_obj, pos_bin, dist_to_bin)
    if dist_to_bin < r:
        # The radius of the bowl is greater than 0.4 but it is hard to grasp at the edge.
        return True
    else:
        return False

def process_data_3d(selected_data, selected_label, with_anchor, task = 'assembly'):
    '''
    This method process the 4-d data(3-d coordinates + 1-d object kind) by:
     1. Balancing the data
     2. Add anchor object information if needed
     3. Add noise to coordinates
     4. Change the index encoding(0 for nut, 1 for bolt) to distributed encodeing(ex. [0.8, 0.2] for nut and [0.9, 0.1] for bolt)
    '''
    u_coord = 0  # The average shift between the approximated coordinates and ground truth
    sigma_coord = 0.006
    u_kind = 0.1
    sigma_kind = 0.05
    u_anchor = 0.1
    sigma_anchor= 0.05

    # selected_data_balanced, selected_label_balanced = balance_data(selected_data, selected_label)
    selected_label_int = label_to_int(selected_label)
    # Object coordinates information
    coord = selected_data[:,0:3]
    obj_kind = selected_data[:, 3]
    noise = np.random.normal(u_coord, sigma_coord, coord.shape)
    coord_noisy = coord + noise
    coord_noisy_balanced, selected_label_balanced = balance_data(coord_noisy, selected_label_int)
    # Object kind information
    kind_onehot = label_to_onehot(obj_kind)
    kind_distributed = onehot_to_distributed(kind_onehot, u_kind, sigma_kind)
    kind_distributed_balanced, selected_label_balanced = balance_data(kind_distributed, selected_label_int)
    # Object anchor information
    anchor_label = get_anchor(selected_label,task)
    anchor_onehot = label_to_onehot(anchor_label)
    anchor_distributed = onehot_to_distributed(anchor_onehot,  u_anchor, sigma_anchor)
    anchor_distributed_balanced, selected_label_balanced = balance_data(anchor_distributed, selected_label_int)

    if with_anchor:
        data_concat = [coord_noisy_balanced, kind_distributed_balanced, anchor_distributed_balanced]
    else:
        data_concat = [coord_noisy_balanced, kind_distributed_balanced]
    return data_concat, selected_label_balanced

def process_batch_data_3d(data, label, with_anchor, task = 'assembly'):
    '''
    This method process the 4-d data(3-d coordinates + 1-d object kind) by:
     1. Balancing the data
     2. Add anchor object information if needed
     3. Add noise to coordinates
     4. Change the index encoding(0 for nut, 1 for bolt) to distributed encodeing(ex. [0.8, 0.2] for nut and [0.9, 0.1] for bolt)
    '''

    for i, _ in enumerate(data):
        selected_data = np.array(list(itertools.chain.from_iterable(data[i])))
        selected_label = np.array(list(itertools.chain.from_iterable(label[i])))
        # selected_data_balanced, selected_label_balanced = balance_data(selected_data, selected_label)
        if 'Noise data' in selected_label:
            continue
        data_concat_balanced, selected_label_balanced = process_data_3d(selected_data, selected_label, with_anchor, task)
        coord_noisy = data_concat_balanced[0]
        kind_distributed = data_concat_balanced[1]


        if with_anchor:
            anchor_distributed = data_concat_balanced[2]
            selected_data_concat = np.concatenate((coord_noisy, kind_distributed, anchor_distributed), axis = 1)
        else:
            selected_data_concat = np.concatenate((coord_noisy, kind_distributed), axis = 1)
        if i == 0:
            data_concat = selected_data_concat
            label_concat = selected_label_balanced
        else:
            data_concat = np.append(data_concat, selected_data_concat, axis = 0)
            label_concat = np.append(label_concat, selected_label_balanced, axis = 0)
    return data_concat, label_concat

def process_data_ssp(selected_data, selected_label, with_anchor, binding, aggregate, dim, wrap_feature = False, task = 'assembly'):
    '''
    '''

    data_concat, selected_label_int = process_data_3d(selected_data, selected_label, with_anchor = True, task = task)
    x_axis_sp = make_good_unitary(dim)
    y_axis_sp = make_good_unitary(dim)
    z_axis_sp = make_good_unitary(dim)

    bolt_sp = make_good_unitary(dim)
    nut_sp = make_good_unitary(dim)

    table_sp = make_good_unitary(dim)
    jig_sp = make_good_unitary(dim)
    bin_origin_sp = make_good_unitary(dim)
    bin_target_sp = make_good_unitary(dim)

    tag1 = make_good_unitary(dim)
    tag2 = make_good_unitary(dim)
    tag3 = make_good_unitary(dim)

    coord_noisy = data_concat[0]
    kind_distributed = data_concat[1]
    anchor_distributed = data_concat[2]
    if task == 'assembly':

        coord_sp = encode_feature(coord_noisy, [x_axis_sp, y_axis_sp, z_axis_sp], binding = binding[0], aggregate = aggregate[0])
        kind_sp = encode_feature(kind_distributed, [nut_sp, bolt_sp], binding = binding[1], aggregate = aggregate[1])
        anchor_sp = encode_feature(anchor_distributed, [table_sp, jig_sp], binding = binding[2], aggregate = aggregate[2])
    elif task == 'bin_picking':
        coord_sp = encode_feature(coord_noisy, [x_axis_sp, y_axis_sp, z_axis_sp], binding = binding[0], aggregate = aggregate[0])
        kind_sp = encode_feature(kind_distributed, [nut_sp, bolt_sp], binding = binding[1], aggregate = aggregate[1])
        anchor_sp = encode_feature(anchor_distributed, [bin_origin_sp, bin_target_sp], binding = binding[2], aggregate = aggregate[2])

    if wrap_feature == True:
        coord_sp = (tag1 * spa.SemanticPointer(coord_sp)).v
        kind_sp = (tag2 * spa.SemanticPointer(kind_sp)).v
        anchor_sp = (tag3 * spa.SemanticPointer(anchor_sp)).v
    if with_anchor:
        data_concat = [coord_sp, kind_sp, anchor_sp]
    else:
        data_concat = [coord_sp, kind_sp]
    return data_concat, selected_label_int

def process_batch_data_ssp(data, label, with_anchor, binding, aggregate, aggregate_between_feature, dim, wrap_feature = False, task = 'assembly'):
    '''

    '''
    x_axis_sp = make_good_unitary(dim)
    y_axis_sp = make_good_unitary(dim)
    z_axis_sp = make_good_unitary(dim)

    bolt_sp = make_good_unitary(dim)
    nut_sp = make_good_unitary(dim)

    table_sp = make_good_unitary(dim)
    jig_sp = make_good_unitary(dim)
    bin_origin_sp = make_good_unitary(dim)
    bin_target_sp = make_good_unitary(dim)

    tag1 = make_good_unitary(dim)
    tag2 = make_good_unitary(dim)
    tag3 = make_good_unitary(dim)

    for i, _ in enumerate(data):
        selected_data = np.array(list(itertools.chain.from_iterable(data[i])))
        selected_label = np.array(list(itertools.chain.from_iterable(label[i])))
        # selected_data_balanced, selected_label_balanced = balance_data(selected_data, selected_label)
        if 'Noise data' in selected_label:
            continue
        selected_data_balanced, selected_label_int_balanced = process_data_3d(selected_data, selected_label, with_anchor = True, task = task)

        coord_noisy = selected_data_balanced[0]
        kind_distributed = selected_data_balanced[1]
        anchor_distributed = selected_data_balanced[2]

        if i == 0:
            coord_noisy_concat = coord_noisy
            kind_distributed_concat = kind_distributed
            anchor_distributed_concat = anchor_distributed
            label_concat = selected_label_int_balanced
        else:
            coord_noisy_concat = np.concatenate((coord_noisy_concat, coord_noisy))
            kind_distributed_concat = np.concatenate((kind_distributed_concat, kind_distributed))
            anchor_distributed_concat = np.concatenate((anchor_distributed_concat, anchor_distributed))
            label_concat = np.concatenate((label_concat, selected_label_int_balanced))
    if task == 'assembly':
        # Anchor object information
        coord_sp = encode_feature(coord_noisy_concat, [x_axis_sp, y_axis_sp, z_axis_sp], binding = binding[0], aggregate = aggregate[0])
        kind_sp = encode_feature(kind_distributed_concat, [nut_sp, bolt_sp], binding = binding[1], aggregate = aggregate[1])
        anchor_sp = encode_feature(anchor_distributed_concat, [table_sp, jig_sp], binding = binding[2], aggregate = aggregate[2])

    elif task == 'bin_picking':
        coord_sp = encode_feature(coord_noisy_concat, [x_axis_sp, y_axis_sp, z_axis_sp], binding = binding[0], aggregate = aggregate[0])
        kind_sp = encode_feature(kind_distributed_concat, [nut_sp, bolt_sp], binding = binding[1], aggregate = aggregate[1])
        anchor_sp = encode_feature(anchor_distributed_concat, [bin_origin_sp, bin_target_sp], binding = binding[2], aggregate = aggregate[2])
    if wrap_feature == True:
        coord_sp = (tag1 * spa.SemanticPointer(coord_sp)).v
        kind_sp = (tag2 * spa.SemanticPointer(kind_sp)).v
        anchor_sp = (tag3 * spa.SemanticPointer(anchor_sp)).v
    if with_anchor:
        data_concat = [coord_sp, kind_sp, anchor_sp]
    else:
        data_concat = [coord_sp, kind_sp]
    data_encoded = encode_dataset(data_concat, aggregate_between_feature)

    return data_encoded, label_concat

def label_to_onehot(x):
    '''
    x: is a 1-d array that contains the labels

    return: an n * d array, n is the number of data. d is the number of classes.
    '''
    n_data = x.shape[0]
    x = [[i] for i in x]
    one_hot = MultiLabelBinarizer()
    result = one_hot.fit_transform(x)
    return result

def onehot_to_distributed(x, u, sigma):
    '''
    x: an n * d array, n is the number of data. d is the number of classes.
    u: is the mean of the noise
    sigma: is the variance of he noise

    return: an n * d array, n is the number of data. d is the number of classes.
    '''
    n_data = x.shape[0]
    n_classes = x.shape[1]
    noise = np.random.normal(u, sigma, n_data)
    x_noisy = abs(x - np.column_stack((noise, noise)))
    return x_noisy

def get_anchor(selected_label_balanced, task = 'assembly'):
    '''
    Get anchor information based on the label information.
    '''
    n_data = selected_label_balanced.shape[0]
    result = np.zeros(n_data)
    if task == 'assembly':
        # Anchor object information
        ind_table = np.concatenate((np.where(selected_label_balanced == 'Nut on table')[0], np.where(selected_label_balanced == 'Bolt on table')[0]))
        ind_jig = np.delete(np.arange(n_data), ind_table)
        result[ind_table] = 0
        result[ind_jig] = 1

    elif task == 'bin_picking':
        ind_bin_origin = np.where(selected_label_balanced == 'Object in origin bin')[0]
        ind_bin_target = np.delete(np.arange(n_data), ind_bin_origin)
        result[ind_bin_origin] = 0
        result[ind_bin_target] = 1
    return result
