from __future__ import division

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
grammar_weights = "../../eq_vae_grammar_h100_c234_L25_E50_batchB.hdf5" #weight_files/zinc_vae_L56.hdf5"
grammar_model = equation_vae.EquationGrammarModel(grammar_weights,latent_rep_size=25)

#import image
import copy
import time
#import networkx as nx

WORST = 1000

MAX = 100000

targets = []
for i in range(len(eqs[:MAX])):
    #targets.append(np.mean(np.minimum(np.abs(np.array(eval(eqs[i])) - y),100)))
    try:
        #score = np.log(1+np.mean(np.minimum(np.abs(np.array(eval(eqs[i])) - y)**2,WORST)))
        score = np.log(1+np.mean(np.minimum((np.array(eval(eqs[i])) - y)**2, WORST)))
    except:
        score = np.log(1+WORST)
    if not np.isfinite(score):
        score = np.log(1+WORST)
    print i, eqs[i], score
    targets.append(score)

targets = np.array(targets)
#pdb.set_trace()

np.savetxt('latent_features_and_targets/targets_eq.txt', targets)
np.savetxt('latent_features_and_targets/x_eq.txt', x)
np.savetxt('latent_features_and_targets/true_y_eq.txt', y)

batch_size = 5000
num_batches = int(np.ceil(len(targets) / batch_size))
for i in xrange(num_batches):
    if i == 0:
        latent_points = grammar_model.encode(eqs[:batch_size])
    else:
        new_latents = grammar_model.encode(eqs[i*batch_size:(i+1)*batch_size])
        latent_points = np.concatenate((latent_points, new_latents), axis=0)

    # We store the results
    np.savetxt('latent_features_and_targets/latent_features_eq.txt', latent_points)



