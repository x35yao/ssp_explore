from encoding_ssp import test_ssp

test_cases = [['False', ['power', 'multiply', 'multiply'],['multiply', 'sum', 'sum'], 'multiply', 'False' ],\
['True', ['power', 'multiply', 'multiply'],['multiply', 'sum', 'sum'], 'multiply', 'False' ],\
['False', ['power', 'multiply', 'multiply'],['multiply', 'sum', 'sum'], 'sum', 'False' ],\
['True', ['power', 'multiply', 'multiply'],['multiply', 'sum', 'sum'], 'sum', 'False' ],\
['False', ['power', 'power', 'power'],['multiply', 'multiply', 'multiply'], 'multiply', 'False' ],\
['True', ['power', 'power', 'power'],['multiply', 'multiply', 'multiply'], 'multiply', 'False' ],\
['False', ['power', 'power', 'power'],['multiply', 'multiply', 'multiply'], 'sum', 'False' ],\
['True', ['power', 'power', 'power'],['multiply', 'multiply', 'multiply'], 'sum', 'False' ],\
['False', ['power', 'multiply', 'multiply'],['multiply', 'sum', 'sum'], 'sum', 'True' ],\
['True', ['power', 'multiply', 'multiply'],['multiply', 'sum', 'sum'], 'sum', 'True' ]]

affinity = ['cosine']
algorithm = 'kmeans'
dim = 256
task = 'assembly'
for i, test_case in enumerate(test_cases):
    with_anchor = test_case[0]
    binding = test_case[1]
    aggregate = test_case[2]
    aggregate_between_feature = test_case[3]
    wrap_feature = test_case[4]
    print(f'Test_case {i}. with_anchor: {with_anchor}, binding: {binding}, aggregate: {aggregate}, aggregate_between_feature: {aggregate_between_feature}, \
    wrap_feature: {wrap_feature}' )
    test_ssp(task, algorithm, with_anchor, affinity, binding, aggregate, aggregate_between_feature, dim, wrap_feature)
