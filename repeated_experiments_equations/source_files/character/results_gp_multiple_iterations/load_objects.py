
import pickle
import gzip

# We define the functions used to load and save objects

def save_object(obj, filename):

    """
    Function that saves an object to a file using pickle
    """

    result = pickle.dumps(obj)
    with gzip.GzipFile(filename, 'wb') as dest: dest.write(result)
    dest.close()


def load_object(filename):

    """
    Function that loads an object from a file using pickle
    """

    with gzip.GzipFile(filename, 'rb') as source: result = source.read()
    ret = pickle.loads(result)
    source.close()

    return ret

iteration = load_object('iteration.dat')
#iteration = 10
best_value = 1e10
n_valid = 0
max_value = 0
for i in range(iteration):
    smiles = load_object('valid_eq_{}.dat'.format(i))
    scores = load_object('scores_eq_{}.dat'.format(i))
    n_valid += len([ x for x in smiles if x is not None ])

    if min(scores) < best_value:
        best_value = min(scores)
    if max(scores) > max_value:
        max_value = max(scores)

import numpy as np

sum_values = 0
count_values = 0
for i in range(iteration):
    scores = np.array(load_object('scores_eq_{}.dat'.format(i)))
    sum_values += np.sum(scores[  scores < max_value ])
    count_values += len(scores[  scores < max_value ])
    
print(1.0 * n_valid / (iteration * 50))
print(best_value)
print(1.0 * sum_values / count_values)
