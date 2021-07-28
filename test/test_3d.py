from encoding_3d import test_3d
import pickle

algorithms = ['agg', 'kmeans', 'gaussian']
with_anchors = ['Ture', 'False']
affinities = [['p_norm'], ['euclidean'], ['cosine'], ['p_norm', 'euclidean', 'euclidean']]

result = []
for algorithm in algorithms:
    for with_anchor in with_anchors:
        for affinity in affinities:
            rand_score, success_rate = test_3d(algorithm, with_anchor, affinity)
            result.append([algorithm, with_anchor, affinity, rand_score, success_rate])
print(result)
with open('test_3d.txt', 'wb') as fp:
    pickle.dump(result, fp)
