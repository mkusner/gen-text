from __future__ import division

import pickle
import gzip
import pdb
import copy
import time


import sys


if __name__ == '__main__':
    try:
        SEED = int(sys.argv[1])
    except:
        raise Exception("First argument must be random number seed")



# sys.path.insert(0, '../')
# import molecule_vae
sys.path.insert(0, '../../')
import equation_vae
diff_model = equation_vae.EquationGrammarModel("../../eq_vae_grammar_h100_c234_L25_E50_batchB.hdf5",25)

def decode_from_latent_space(latent_points, vae_model):

    decode_attempts = 500
    decoded_equations = []
    for i in range(decode_attempts):
        decoded_equations.append(vae_model.decode(latent_points))

    # We see which ones parse according to the grammar
    x = 0 # make x a dummy variable
    parsed_equations = []
    for i in range(decode_attempts):
        parsed_equations.append([])
        for j in range(latent_points.shape[ 0 ]):
            eqn = np.array([ decoded_equations[ i ][ j ] ]).astype('str')[ 0 ]
            tokens = equation_vae.tokenize(eqn)
            try:
                parse = diff_model._parser.parse(tokens).next()
                parsed_equations[ i ].append(eqn)
            except:
                parsed_equations[i].append(None)

    import collections

    decoded_equations = np.array(decoded_equations)
    parsed_equations = np.array(parsed_equations)

    final_eqns = []
    for i in range(latent_points.shape[ 0 ]):

        aux = collections.Counter(parsed_equations[ ~np.equal(parsed_equations[ :, i ], None) , i ])
        if len(aux) > 0:
            eqn = aux.items()[ np.argmax(aux.values()) ][ 0 ]
        else:
            eqn = None
        final_eqns.append(eqn)

    return final_eqns

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

from sparse_gp import SparseGP

import scipy.stats    as sps

import numpy as np
from numpy import * 



np.random.seed(SEED)

# We load the data

# X = np.loadtxt('latent_features_and_targets/latent_faetures.txt')
# y = -np.loadtxt('latent_features_and_targets/targets.txt')
# y = y.reshape((-1, 1))
X = np.loadtxt('latent_features_and_targets/latent_features_eq.txt')
y = np.loadtxt('latent_features_and_targets/targets_eq.txt')
y = y.reshape((-1, 1))



n = X.shape[ 0 ]
permutation = np.random.choice(n, n, replace = False)

X_train = X[ permutation, : ][ 0 : np.int(np.round(0.9 * n)), : ]
X_test = X[ permutation, : ][ np.int(np.round(0.9 * n)) :, : ]

y_train = y[ permutation ][ 0 : np.int(np.round(0.9 * n)) ]
y_test = y[ permutation ][ np.int(np.round(0.9 * n)) : ]

M = 500
sgp = SparseGP(X_train, 0 * X_train, y_train, M)


for iteration in range(100):


    # We fit the GP

    sgp.n_points = X_train.shape[ 0 ]
    sgp.sparse_gp.n_points = X_train.shape[ 0 ]
    sgp.train_via_ADAM(X_train, 0 * X_train, y_train, X_test, X_test * 0,  \
        y_test, minibatch_size = 10 * M, max_iterations = 50, learning_rate = 0.0005)

    # We load some previous trained gp

    pred, uncert = sgp.predict(X_test, 0 * X_test)
    error = np.sqrt(np.mean((pred - y_test)**2))
    testll = np.mean(sps.norm.logpdf(pred - y_test, scale = np.sqrt(uncert)))
    print 'Test RMSE: ', error
    print 'Test ll: ', testll

    pred, uncert = sgp.predict(X_train, 0 * X_train)
    error = np.sqrt(np.mean((pred - y_train)**2))
    trainll = np.mean(sps.norm.logpdf(pred - y_train, scale = np.sqrt(uncert)))
    print 'Test RMSE: ', error
    print 'Test ll: ', trainll

    # We load the decoder to obtain the equations

    char_weights = "../../eq_vae_str_h100_c234_L25_E50_batchB.hdf5"
    char_model = equation_vae.EquationCharacterModel(char_weights,latent_rep_size=25)


    # We pick the next 50 inputs

    next_inputs = sgp.batched_greedy_ei(50, np.min(X_train, 0), np.max(X_train, 0))

    valid_eq_final = decode_from_latent_space(next_inputs, char_model)

    valid_eq_final_final = []
    new_features = []
    for i in range(len(valid_eq_final)):
        if valid_eq_final[ i ] is not None:
            valid_eq_final_final.append(valid_eq_final[ i ])
            new_features.append(next_inputs[ i, : ])
    new_features = np.array(new_features)
    valid_eq_final = valid_eq_final_final

    save_object(valid_eq_final, "results_gp_multiple_iterations/valid_eq_parse_%d.seed_%d.dat" % (iteration, SEED))


    x = np.loadtxt('latent_features_and_targets/x_eq.txt')
    y = np.loadtxt('latent_features_and_targets/true_y_eq.txt')

    WORST = 1000
    scores = []
    for i in range(len(valid_eq_final)):
        
#         score = np.mean(np.minimum(np.abs(np.array(eval(valid_eq_final[i])) - y)**2,100))
        try:
            #score = np.log(1+np.mean((np.array(eval(valid_eq_final[i])) - y)**2))
            score = np.log(1+np.mean(np.minimum((np.array(eval(valid_eq_final[i])) - y)**2, WORST)))
        except:
            score = np.log(1+WORST)

        scores.append(score)
        print(i)
#     print(valid_eq_final)
#     print(scores)

    save_object(scores, "results_gp_multiple_iterations/scores_parse_%d.seed_%d.dat" % (iteration, SEED))
    if len(new_features) > 0:
        X_train = np.concatenate([ X_train, new_features ], 0)
        y_train = np.concatenate([ y_train, np.array(scores)[ :, None ] ], 0)
