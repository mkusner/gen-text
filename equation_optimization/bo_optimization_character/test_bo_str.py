
import pickle
import gzip
import pdb

def decode_from_latent_space(latent_points, grammar_model):

    decode_attempts = 500
    decoded_molecules = []
    for i in range(decode_attempts):
        decoded_molecules.append(grammar_model.decode(latent_points))

    # We see which ones are decoded by rdkit
    x = 0 # make x a dummy variable
    rdkit_molecules = []
    for i in range(decode_attempts):
        rdkit_molecules.append([])
        for j in range(latent_points.shape[ 0 ]):
            smile = np.array([ decoded_molecules[ i ][ j ] ]).astype('str')[ 0 ]
            try:
                res = eval(smile)
                rdkit_molecules[ i ].append(smile)
            except:
                rdkit_molecules[i].append(None)

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

from sparse_gp import SparseGP

import scipy.stats    as sps

import numpy as np
from numpy import * 

np.random.seed(1)

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

for iteration in range(100):

    # We fit the GP
    M = 500
    sgp = SparseGP(X_train, 0 * X_train, y_train, M)
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

    # from rdkit.Chem import MolFromSmiles, MolToSmiles
    # from rdkit.Chem import Draw
    # import image
    import copy
    import time

    import sys
    # sys.path.insert(0, '../')
    # import molecule_vae
    sys.path.insert(0, '../../')
    import equation_vae
    # grammar_weights = "../weight_files/zinc_vae_L56.hdf5"
    # grammar_model = molecule_vae.ZincGrammarModel(grammar_weights)
    grammar_weights = "../../eq_vae_h50_c123_str_L10.hdf5" #weight_files/zinc_vae_L56.hdf5"
    grammar_model = equation_vae.EquationCharacterModel(grammar_weights,latent_rep_size=10)


    # We pick the next 50 inputs

    next_inputs = sgp.batched_greedy_ei(50, np.min(X_train, 0), np.max(X_train, 0))

    valid_eq_final = decode_from_latent_space(next_inputs, grammar_model)

    # from rdkit.Chem import Descriptors
    # from rdkit.Chem import MolFromSmiles, MolToSmiles

    valid_eq_final_final = []
    new_features = []
    for i in range(len(valid_eq_final)):
        if valid_eq_final[ i ] is not None:
            valid_eq_final_final.append(valid_eq_final[ i ])
            new_features.append(next_inputs[ i, : ])
    new_features = np.array(new_features)
    valid_eq_final = valid_eq_final_final

    save_object(valid_eq_final, "results_gp_multiple_iterations/valid_eq{}.dat".format(iteration))

    # logP_values = np.loadtxt('latent_features_and_targets/logP_values.txt')
    # SA_scores = np.loadtxt('latent_features_and_targets/SA_scores.txt')
    # cycle_scores = np.loadtxt('latent_features_and_targets/cycle_scores.txt')
    # SA_scores_normalized = (np.array(SA_scores) - np.mean(SA_scores)) / np.std(SA_scores)
    # logP_values_normalized = (np.array(logP_values) - np.mean(logP_values)) / np.std(logP_values)
    # cycle_scores_normalized = (np.array(cycle_scores) - np.mean(cycle_scores)) / np.std(cycle_scores)

    # targets = SA_scores_normalized + logP_values_normalized + cycle_scores_normalized

    # import sascorer
    # import networkx as nx
    # from rdkit.Chem import rdmolops
    x = np.loadtxt('latent_features_and_targets/x_eq.txt')
    y = np.loadtxt('latent_features_and_targets/true_y_eq.txt')

    scores = []
    for i in range(len(valid_eq_final)):
        
        score = np.mean(np.minimum(np.abs(np.array(eval(valid_eq_final[i])) - y)**2,100))
        # current_log_P_value = Descriptors.MolLogP(MolFromSmiles(valid_smiles_final[ i ]))
        # current_SA_score = -sascorer.calculateScore(MolFromSmiles(valid_smiles_final[ i ]))
        # cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(MolFromSmiles(valid_smiles_final[ i ]))))
        # if len(cycle_list) == 0:
        #     cycle_length = 0
        # else:
        #     cycle_length = max([ len(j) for j in cycle_list ])
        # if cycle_length <= 6:
        #     cycle_length = 0
        # else:
        #     cycle_length = cycle_length - 6

        # current_cycle_score = -cycle_length
     
        # current_SA_score_normalized = (current_SA_score - np.mean(SA_scores)) / np.std(SA_scores)
        # current_log_P_value_normalized = (current_log_P_value - np.mean(logP_values)) / np.std(logP_values)
        # current_cycle_score_normalized = (current_cycle_score - np.mean(cycle_scores)) / np.std(cycle_scores)

        # score = (current_SA_score_normalized + current_log_P_value_normalized + current_cycle_score_normalized)
        # eval(valid_)
        # scores.append(-score)
        scores.append(score)
        print(i)
    print(valid_eq_final)
    print(scores)

    save_object(scores, "results_gp_multiple_iterations/scores{}.dat".format(iteration))

    X_train = np.concatenate([ X_train, new_features ], 0)
    y_train = np.concatenate([ y_train, np.array(scores)[ :, None ] ], 0)
