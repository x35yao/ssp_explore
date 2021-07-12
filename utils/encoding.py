import numpy as np
import nengo.spa as spa

def power(s, e):
    x = np.fft.ifft(np.fft.fft(s.v) ** e).real
    return spa.SemanticPointer(data=x)

def make_good_unitary(D, eps=1e-3, rng=np.random):
    a = rng.rand((D - 1) // 2)
    sign = rng.choice((-1, +1), len(a))
    phi = sign * np.pi * (eps + a * (1 - 2 * eps))
    assert np.all(np.abs(phi) >= np.pi * eps)
    assert np.all(np.abs(phi) <= np.pi * (1 - eps))

    fv = np.zeros(D, dtype='complex64')
    fv[0] = 1
    fv[1:(D + 1) // 2] = np.cos(phi) + 1j * np.sin(phi)
    fv[-1:D // 2:-1] = np.conj(fv[1:(D + 1) // 2])
    if D % 2 == 0:
        fv[D // 2] = 1

    assert np.allclose(np.abs(fv), 1)
    v = np.fft.ifft(fv)
    # assert np.allclose(v.imag, 0, atol=1e-5)
    v = v.real
    assert np.allclose(np.fft.fft(v), fv)
    assert np.allclose(np.linalg.norm(v), 1)
    return spa.SemanticPointer(v)

def encode_point(x, y, z, x_axis_sp, y_axis_sp, z_axis_sp):

    return power(x_axis_sp, x) * power(y_axis_sp, y) * power(z_axis_sp, z)

def encode_entry(x, sp, binding = 'multiply'):
    if binding == 'multiply':
        result = x * sp
    elif binding == 'power':
        result = power(sp, x)
    return result

def encode_datapoint(datapoint, sps, binding = 'multiply', aggregate = 'sum'):
    '''
    datapoint: an 1 by m array. m is the number of possible choices of the property being encoded.
          (ex. when encoding object kind, the possible choices of object kind are 'bolt' and 'nut').
    sps: A list of size m. Each entry is the semantic pointer for each of the possible choices.
    binding: The type to bind each entry. There are two ways:
          1. 'multiply': c1*sp1
          2. 'power': sp1^c1
    aggregate: The type to bind between entries. There are two ways:
          1. 'sum': c1*sp1 + c2 * sp2
          2. 'convolution': sp1^c1 * sp2^c2
    '''
    m = len(sps)
    if aggregate == 'sum':
        result = 0
        for i in range(m):
            result += encode_entry(datapoint[i], sps[i], binding = binding).v
        return result
    elif aggregate == 'convolution':
        result = 1
        for i in range(m):
            result *= encode_entry(datapoint[i], sps[i], binding = binding)
        return result.v

def encode_feature(dataset, sps, binding = 'multiply', aggregate = 'sum'):
    n = dataset.shape[0]
    dim = sps[0].v.shape[0]
    result = np.zeros((n, dim))
    for i in range(n):
        result[i, :] = encode_datapoint(dataset[i], sps, binding, aggregate)
    return result

def encode_dataset(encoded_feature, aggregate_between_feature = 'sum'):
    '''
    encoded_feature: A list of encoded_features. Each entry is an numpy array with size n * dim.
    aggregate_between_feature:
    '''
    m = len(encoded_feature)
    n = encoded_feature[0].shape[0]
    dim = encoded_feature[1].shape[1]
    result = np.zeros((n, dim))
    if aggregate_between_feature == 'sum':
        for i in range(n):
            for j in range(m):
                result[i,:] += encoded_feature[j][i,:]
    elif aggregate_between_feature == 'convolution':
        temp = 1
        for i in range(n):
            for j in range(m):
                temp *= spa.SemanticPointer(data=encoded_feature[j][i,:])
            result[i,:] = temp.v
    return result
