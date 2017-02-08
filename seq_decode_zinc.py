from __future__ import print_function

import argparse
import os
import h5py
import numpy as np
import sys
import nltk
import six
import cPickle as pickle

from molecules.model_gr_prev import MoleculeVAE
from molecules.utils import one_hot_array, one_hot_index, from_one_hot_array, \
    decode_smiles_from_indexes, load_dataset

from molecules.utils import many_one_hot
#from pylab import figure, axes, scatter, title, show

#from rdkit import Chem
#from rdkit.Chem import Draw

import pdb
import zinc_grammar as G
#from rdkit import Chem

from sklearn.metrics.pairwise import euclidean_distances

from keras import backend as K

rules = G.gram.split('\n')
productions = G.GCFG.productions()

MAX_LEN = 277
DIM = len(rules)
LATENT = 2 #292
EPOCHS = 20
BATCH = 500


def prod_to_string(P,string):
    if len(P) == 0:
        return string
    tup = P[0].rhs()
    for item in tup:
        if len(P) == 0:
            return string
        if isinstance(item,six.string_types):
            string = string + item
        else:
            P.pop(0)
            string = prod_to_string(P, string)
    return string


def get_strings2(INDS): # (10, MAX_LEN, DIM)

    sn_rules = []
    for i in range(INDS.shape[0]): #s in sn:
        rule_list = []
        for r in range(MAX_LEN):
            ix = int(np.where(INDS[i,r,:] == 1)[0][0])
            rule_list.append(productions[ix])
        sn_rules.append(prod_to_string(rule_list,''))

    return np.array(sn_rules)



def prods_to_eq(prods):
    seq = [prods[0].lhs()]
    for prod in prods:
        if str(prod.lhs()) == 'Q':
            break
        for ix, s in enumerate(seq):
            if s == prod.lhs():
                seq = seq[:ix] + list(prod.rhs()) + seq[ix+1:]
                break
    return ''.join(seq)

def get_stringsB(INDS):
    n = INDS.shape[0]

    examples = [[productions[INDS[index,t].argmax()] for t in xrange(MAX_LEN)] for index in xrange(n)]
    #for ix, eq in enumerate(raw[:batch_size]):
    stringsB = []
    for i in range(len(examples)):
        try:
            stringsB.append(prods_to_eq(examples[i]))
        except:
            stringsB.append('invalid')
            #print "*invalid*"
    return stringsB


def hard_sigmoid(x):
    x = 0.2*x + 0.5
    return np.clip(x, 0.0, 1.0)

class SeqDecode():

    # call 2nd
    def setup(self, model, on_gpu):
        self.model = model
    
        inp = self.model.decoder.layers[0].input
        next_to_last = self.model.decoder.layers[-2].output
        self.get_output_from_input = K.function([inp],[next_to_last])

        if on_gpu == 1:
            raise ValueError('not yet!')
        else:
            self.W_z = K.eval(self.model.decoder.layers[-1].W_z)
            self.W_h = K.eval(self.model.decoder.layers[-1].W_h)
            self.W_r = K.eval(self.model.decoder.layers[-1].W_r)
            self.b_z = K.eval(self.model.decoder.layers[-1].b_z)
            self.b_h = K.eval(self.model.decoder.layers[-1].b_h)
            self.b_r = K.eval(self.model.decoder.layers[-1].b_r)
            self.U_h = K.eval(self.model.decoder.layers[-1].U_h)
            self.U_z = K.eval(self.model.decoder.layers[-1].U_z)
            self.U_r = K.eval(self.model.decoder.layers[-1].U_r)
            self.Y = K.eval(self.model.decoder.layers[-1].Y)


    
    #shape = (100,MAX_LEN+1)
    
    def cond_sample_np(self, x, STACK, POINT):
        shape = x.shape # (n,d)
        samples = np.zeros((shape[0],shape[1]))
        for i in range(shape[0]):
            if POINT[i] == -1: # this check needs to be done in keras, this means we are done
                samples[i,-1] = 1
                continue
            # 1. pop current nt off stack
            current_nt = STACK[i,POINT[i]]
            POINT[i] = POINT[i]-1 
            the_mask = G.masks[current_nt]
            #where_zero = np.where(the_mask == 0)[0]
            #where = tf.equal(the_mask, zero)
            #the_mask[the_mask == 0] = -999 # hack to deal with gumbel noise making things negative
            #the_mask[the_mask == 1] = 0
            masked = x[i] * the_mask
            masked[masked == 0] = -999
        
            #softmax = masked / K.sum(masked, axis=-1)
            # find tensorflow code for discrete distribution sampling or do gumbel trick
            GU = np.random.gumbel(size=x[i].shape)
            noise_masked = GU + masked
            choice = np.argmax(noise_masked)
            
            # instead of making 1-hot, just put 1 in right place
            # (if stack is empty then we will place nothing
            # then we will look for all zero columns and put 1's at end, either in Keras or as post-processing        
            samples[i,choice] = 1
            
            #new_nts = np.where(rhs_map_sparse[choice] == 1)[0]
            
            
            new_nts = G.rhs_map[choice]
            len_nts = len(new_nts)
            if len_nts == 0:
                continue

            STACK[i,POINT[i]+1:POINT[i]+1+len_nts] = np.flipud(new_nts)
            POINT[i] = POINT[i]+len_nts
            # need to flip
        return (samples, STACK, POINT)





    
    def numpy_decode(self, model, latent, on_gpu):
        # model - keras model
        # latent - numpy output from encoder
        #decoder = model.decoder
        #decoder.layers = decoder.layers[:-1]
    
        batch_size = latent.shape[0]
        STACK = np.zeros((batch_size,(MAX_LEN*(G.max_rhs-1)+1)),dtype=np.int32)
        POINT = np.zeros((batch_size,),dtype=np.int32)
    
        final_layer_input = self.get_output_from_input([latent])[0]     # (batch, max_len, D)
            
        if on_gpu == 1:
            raise ValueError('not yet!')
        else:
            x_z = np.dot(final_layer_input, self.W_z) + self.b_z
            x_h = np.dot(final_layer_input, self.W_h) + self.b_h
            x_r = np.dot(final_layer_input, self.W_r) + self.b_r
            
            h = np.zeros((batch_size, x_z.shape[-1]))
            H =  np.zeros((batch_size, MAX_LEN, DIM))
            y = np.zeros((batch_size, x_z.shape[-1]))
            X_hat = np.zeros((batch_size, MAX_LEN, x_z.shape[-1]))
            softmask = np.zeros((batch_size, MAX_LEN, DIM))
    
            for t in range(MAX_LEN):
                z = hard_sigmoid(x_z[:,t] + np.dot(h, self.U_z))
                r = hard_sigmoid(x_r[:,t] + np.dot(h, self.U_r))
                hh = np.tanh(x_h[:,t] + np.dot(r*h, self.U_h) + np.dot(r*y, self.Y))
                h = z*h + (1-z)*hh
                (y, STACK, POINT) = self.cond_sample_np(h, STACK, POINT)
                #softmask[:,t,:] = mask_training(h, data[:,t,:])
                H[:,t,:] = h 
    
                X_hat[:,t,:] = y
        return X_hat


