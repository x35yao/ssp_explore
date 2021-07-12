import numpy as np
import pickle
from scipy.stats import trim_mean
from sklearn.covariance import MinCovDet

def fetch_data(x, force_success = True):
    '''
    This function returns the coordinates of bolts and nuts with NO labels.

    x: a list of file_path
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
    if force_success:
        for seq in seqs:
            if len(seq) == 4:
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

def fetch_data_with_label(x, force_success = True):
    '''
    x: a list of file_path
    force_success: will only take the data from successful attempt which will then reduce the number of data but also balance the data
    return: x_nut, y_nut, x_bolt, y_bolt
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
    if force_success:
        for seq in seqs:
            if len(seq) == 4:
                seqs_success.append(seq)
        seqs = seqs_success
    test_nut = []
    test_bolt = []
    for seq in seqs:
        for step in seq:
            if step['action'] == 'PUT_NUT_IN_JIG':
                data = [obj['pos'] for obj in step['objs'] if obj['class'] == 'nut'][0]
                test_nut.append(list(data) + ['Nut on table'])
            elif step['action'] == 'PUT_BOLT_IN_JIG':
                data = [obj['pos'] for obj in step['objs'] if obj['class'] == 'bolt'][0]
                if data[1] < 0.03:
                    test_bolt.append(list(data) + ['Bolt on table'])
            elif step['action'] == 'ASSEMBLE':
                data_nut = [obj['pos'] for obj in step['objs'] if obj['class'] == 'nut'][0]
                if data_nut[1] < 0.02: # Wrong labeld data
                    pass
                else:
                    test_nut.append(list(data_nut) + ['Nut in jig'])
                data_bolt = [obj['pos'] for obj in step['objs'] if obj['class'] == 'bolt'][0]
                if data_bolt[1] < 0.06: # Wrong labeld data
                    pass
                else:
                    test_bolt.append(list(data_bolt) + ['Bolt in jig'])
            elif step['action'] == 'ASSEMBLED' or step['action'] == None:
                data = [obj['pos'] for obj in step['objs'] if obj['class'] == 'bolt'][0]
                # test_bolt.append(list(data) + ['Bolt assembled'])
                if data[2] > -0.6:
                    pass
                else:
                    test_bolt.append(list(data) + ['Bolt assembled'])
    data_nut = np.array(test_nut)
    data_bolt = np.array(test_bolt)
    x_nut = data_nut[:,:3].astype(np.float)
    y_nut = data_nut[:,3]
    x_bolt = data_bolt[:,:3].astype(np.float)
    y_bolt = data_bolt[:,3]
    # Seperate data a little more
    delta = 0.01
    for i, val in enumerate(x_bolt[:,1]):
        if val > 0.064:
            x_bolt[i][1] += delta
    return x_nut, y_nut, x_bolt, y_bolt

def fetch_data_with_label_per_step(x, force_success = True):
    '''
    x: a list of file_path
    force_success: will only take the data from successful attempt which will then reduce the number of data but also balance the data

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
    if force_success:
        for seq in seqs:
            if len(seq) == 4:
                seqs_success.append(seq)
        seqs = seqs_success
    data_concat = []
    label = []
    for j, seq in enumerate(seqs):
        data_concat_seq = []
        label_seq = []
        for step in seq:
            if step['action'] == 'PUT_NUT_IN_JIG':
                label_nut = []
                label_bolt = []
                data_nut = [list(obj['pos']) + [0] for obj in step['objs'] if obj['class'] == 'nut']
                data_bolt = [list(obj['pos']) + [1] for obj in step['objs'] if obj['class'] == 'bolt']
                data_concat_seq.append(data_nut + data_bolt)
                for i in data_nut:
                    label_nut.append('Nut on table')
                for i in data_bolt:
                    label_bolt.append('Bolt on table')
                label_seq.append(label_nut + label_bolt)
            elif step['action'] == 'PUT_BOLT_IN_JIG':
                label_nut = []
                label_bolt = []
                data_nut = [list(obj['pos']) + [0] for obj in step['objs'] if obj['class'] == 'nut']
                data_bolt = [list(obj['pos']) + [1] for obj in step['objs'] if obj['class'] == 'bolt']
                data_concat_seq.append(data_nut + data_bolt)
                for i in data_nut:  # Skechy here, some trial starts from 'Put bolt in jig' and don't have 'Put nut in jig'. Might be troublesome.
                    label_nut.append('Nut on table')
                for i in data_bolt:
                    label_bolt.append('Bolt on table')
                if data_nut[0][1] > 0.04:
                    label_nut[0] = 'Nut in jig'
                label_seq.append(label_nut + label_bolt)
            elif step['action'] == 'ASSEMBLE':
                label_nut = []
                label_bolt = []
                data_nut = [list(obj['pos']) + [0] for obj in step['objs'] if obj['class'] == 'nut']
                data_bolt = [list(obj['pos']) + [1] for obj in step['objs'] if obj['class'] == 'bolt']
                data_concat_seq.append(data_nut + data_bolt)
                for i in data_nut:
                    label_nut.append('Nut on table')
                for i in data_bolt:
                    label_bolt.append('Bolt on table')
                if data_nut[0][1] > 0.04:
                    label_nut[0] = 'Nut in jig'
                if data_bolt[0][1] > 0.06:
                    label_bolt[0] = 'Bolt in jig'
                label_seq.append(label_nut + label_bolt)

            elif step['action'] == 'ASSEMBLED' or step['action'] == None:
                label_nut = []
                label_bolt = []
                data_nut = [list(obj['pos']) + [0] for obj in step['objs'] if obj['class'] == 'nut']
                data_bolt = [list(obj['pos']) + [1] for obj in step['objs'] if obj['class'] == 'bolt']
                data_concat_seq.append(data_nut + data_bolt)
                for i in data_nut:  # Skechy here, some trial starts from 'Put bolt in jig' and don't have 'Put nut in jig'. Might be troublesome.
                    label_nut.append('Nut on table')
                for i in data_bolt:
                    label_bolt.append('Bolt on table')
                if data_nut[0][1] > 0.04:
                    label_nut[0] = 'Nut in jig'
                if data_bolt[0][1] > 0.06:
                    label_bolt[0] = 'Bolt assembled'
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

def divide_data(x, y, n_samples):
    '''
    Divide data into training set and test set.
    x: all the data with shape n_data by n_dimension
    y: lables for all the data with n_data entries
    n_samples: number of samples for training set

    return:
    x_train, y_train, x_test, y_test
    '''
    n_data = x.shape[0]
    n_training = len(set(y)) * n_samples
    n_test = n_data - n_training
    x_train = np.empty((n_training, 3))
    y_train = np.empty(n_training, dtype = y.dtype)
    x_test = np.empty((n_test,3))
    y_test = np.empty(n_test, dtype = y.dtype)

    train_inds = []
    for i, label in enumerate(set(y)):
        ind = np.where(y == label)[0]
        selected_ind = np.random.choice(ind, n_samples, replace = False)
        train_inds += selected_ind.tolist()
        x_train[i * n_samples:(i + 1) * n_samples,:] = x[selected_ind, :]
        y_train[i * n_samples:(i + 1) * n_samples] = y[selected_ind]
    inds = np.arange(n_data)
    test_inds = np.delete(inds, np.array(train_inds))
    x_test = x[test_inds,:]
    y_test = y[test_inds]

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
