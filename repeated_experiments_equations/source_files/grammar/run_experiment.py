
import argparse
import pickle
import gzip
import pdb

import sys
sys.path.insert(0, '../../../')
import equation_vae
from numpy import * # need this for evaluating equations
from sparse_gp import SparseGP
import scipy.stats    as sps
import numpy as np
import os.path
import os
import copy
import time



def get_arguments():
    parser = argparse.ArgumentParser(description='BO experiment')
    parser.add_argument('--exp_seed', type=int, metavar='N',
                        help='Experiment seed.')
    return parser.parse_args()

def decode_from_latent_space(latent_points, grammar_model):

    decode_attempts = 25
    decoded_molecules = []
    for i in range(decode_attempts):
        current_decoded_molecules = grammar_model.decode(latent_points)
        #current_decoded_molecules = [ x if x != '' else 'Sequence too long' for x in current_decoded_molecules ]
        decoded_molecules.append(current_decoded_molecules)

    # We see which ones are decoded by rdkit
    
    rdkit_molecules = []
    for i in range(decode_attempts):
        rdkit_molecules.append([])
        for j in range(latent_points.shape[ 0 ]):
            smile = np.array([ decoded_molecules[ i ][ j ] ]).astype('str')[ 0 ]
            if smile == '':
                rdkit_molecules[ i ].append(None)
            else:
                rdkit_molecules[ i ].append(smile)

    import collections

    decoded_molecules = np.array(decoded_molecules)
    rdkit_molecules = np.array(rdkit_molecules)

    final_smiles = []
    for i in range(latent_points.shape[ 0 ]):

        aux = collections.Counter(rdkit_molecules[ ~np.equal(rdkit_molecules[ :, i ], None) , i ])
        if len(aux) > 0:
            smile = aux.items()[ np.argmax(aux.values()) ][ 0 ]
        else:
            smile = None
        final_smiles.append(smile)

    return final_smiles

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




args = get_arguments()
np.random.seed(args.exp_seed)

# Make folder for results
directory = "results_gp_multiple_iterations/"
if not os.path.exists(directory):
    os.makedirs(directory)

# We load the data

X = np.loadtxt('latent_features_and_targets/latent_features_eq.txt')
y = np.loadtxt('latent_features_and_targets/targets_eq.txt')
y = y.reshape((-1, 1))

n = X.shape[ 0 ]
permutation = np.random.choice(n, n, replace = False)

X_train = X[ permutation, : ][ 0 : np.int(np.round(0.9 * n)), : ]
X_test = X[ permutation, : ][ np.int(np.round(0.9 * n)) :, : ]

y_train = y[ permutation ][ 0 : np.int(np.round(0.9 * n)) ]
y_test = y[ permutation ][ np.int(np.round(0.9 * n)) : ]

for iteration in range(5):

    # We fit the GP
    
    if iteration == 0 and os.path.isfile(directory + "X_train.dat") and \
        os.path.isfile(directory + "y_train.dat") and os.path.isfile(directory + "iteration.dat"):

        X_train = load_object(directory + "X_train.dat")
        y_train = load_object(directory + "y_train.dat")
        iteration = load_object(directory + "iteration.dat")
    else:
        save_object(X_train, directory + "X_train.dat")
        save_object(y_train, directory + "y_train.dat")
        save_object(iteration, directory + "iteration.dat")

    np.random.seed(args.exp_seed * iteration)
    M = 500
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

    # We load the decoder to obtain the molecules

    # grammar_weights = "../eq_vae_grammar_h100_c234_L25_E50_batchB.hdf5"
    # grammar_model = equation_vae.EquationGrammarModel(grammar_weights,latent_rep_size=25)
    grammar_weights = "../../../eq_vae_grammar_h100_c234_L25_E50_batchB.hdf5" #weight_files/zinc_vae_L56.hdf5"
    grammar_model = equation_vae.EquationGrammarModel(grammar_weights,latent_rep_size=25)
    # We pick the next 50 inputs

    next_inputs = sgp.batched_greedy_ei(50, np.min(X_train, 0), np.max(X_train, 0))

    valid_eq_final = decode_from_latent_space(next_inputs, grammar_model)

#    valid_eq_final_final = []
#    new_features = []
#    for i in range(len(valid_eq_final)):
#        if valid_eq_final[ i ] is not None:
#            valid_eq_final_final.append(valid_eq_final[ i ])
#            new_features.append(next_inputs[ i, : ])
#    new_features = np.array(new_features)
#    valid_eq_final = valid_eq_final_final

    new_features = next_inputs

    save_object(valid_eq_final, directory + "valid_eq_{}.dat".format(iteration))

    x = np.loadtxt('latent_features_and_targets/x_eq.txt')
    yT = np.loadtxt('latent_features_and_targets/true_y_eq.txt')

    scores = []
    WORST = 1000
    for i in range(len(valid_eq_final)):
        if valid_eq_final[ i ] is not None: 
            try:
                score = np.log(1+np.mean(np.minimum((np.array(eval(valid_eq_final[i])) - yT)**2, WORST)))
            except:
                score = np.log(1+WORST)
            if not np.isfinite(score):
                score = np.log(1+WORST)
        else:
            score = np.log(1+WORST)

        scores.append(score)
        print(i)

    print(valid_eq_final)
    print(scores)

    save_object(scores, directory + "scores_eq_{}.dat".format(iteration))

    if len(new_features) > 0:
        X_train = np.concatenate([ X_train, new_features ], 0)
        y_train = np.concatenate([ y_train, np.array(scores)[ :, None ] ], 0)
