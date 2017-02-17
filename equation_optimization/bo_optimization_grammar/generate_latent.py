

#import sascorer

import numpy as np  
import pdb

# We load the smiles data

fname = '../../equation2_15_dataset.txt' #'250k_rndm_zinc_drugs_clean.smi'

from numpy import *
import cPickle as pickle
test = '1 / 3 + x + sin( x * x )'
x = np.linspace(-10,10,1000)
y = np.array(eval(test))
#pickle.dump({'x': x, 'y': y}, open('../../eq_input_output.p','wb'))




with open(fname) as f:
    eqs = f.readlines()

for i in range(len(eqs)):
    eqs[ i ] = eqs[ i ].strip()
    eqs[ i ] = eqs[ i ].replace(' ','')

# We load the auto-encoder

import sys
sys.path.insert(0, '../../')
import equation_vae
grammar_weights = "../../eq_vae_h50_c123_cond_L10.hdf5" #weight_files/zinc_vae_L56.hdf5"
grammar_model = equation_vae.EquationGrammarModel(grammar_weights,latent_rep_size=10)

#import image
import copy
import time
#import networkx as nx

targets = []
for i in range(len(eqs)):
    #targets.append(np.mean(np.minimum(np.abs(np.array(eval(eqs[i])) - y),100)))
    targets.append(np.mean(np.minimum(np.abs(np.array(eval(eqs[i])) - y)**2,100)))

targets = np.array(targets)
#pdb.set_trace()

latent_points = grammar_model.encode(eqs)

# We store the results

latent_points = np.array(latent_points)
np.savetxt('latent_features_and_targets/latent_features_eq.txt', latent_points)
np.savetxt('latent_features_and_targets/targets_eq.txt', targets)
np.savetxt('latent_features_and_targets/x_eq.txt', x)
np.savetxt('latent_features_and_targets/true_y_eq.txt', y)



