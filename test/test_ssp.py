from encoding_ssp import test_ssp
import pickle

algorithms = ['agg', 'kmeans', 'gaussian']
with_anchors = ['Ture', 'False']
affinities = [['p_norm'], ['euclidean'], ['cosine']]

test_cases = {}
# test_1: (c_1 * bolt_ssp + c_2*nut_ssp) * coord_ssp
test_cases['test_1'] = [['power', 'multiply', 'mutiply'], ['convolution', 'sum', 'sum'], 'convolution']
# test_2: (c_1 * bolt_ssp + c_2*nut_ssp) + coord_ssp
test_cases['test_2'] = [['power', 'multiply', 'mutiply'], ['convolution', 'sum', 'sum'], 'sum']
# test_3: bolt_ssp^c1 * nut_ssp^2 * X_sp^x * Y_sp^y * Z_sp^z
test_cases['test_3'] = [['power', 'power', 'power'], ['convolution', 'convolution', 'convolution'], 'convolution']
for algorithm in algorithms:
    for with_anchor in with_anchors:
        for affinity in affinities:
