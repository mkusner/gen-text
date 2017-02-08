from __future__ import print_function

import argparse
import os
import h5py
import numpy as np
import sys
import nltk
import six
import cPickle as pickle

from molecules.model_zinc import MoleculeVAE
from molecules.utils import one_hot_array, one_hot_index, from_one_hot_array, \
    decode_smiles_from_indexes, load_dataset

from molecules.utils import many_one_hot
#from pylab import figure, axes, scatter, title, show

#from rdkit import Chem
#from rdkit.Chem import Draw

import pdb
from scipy.stats import multivariate_normal
import zinc_grammar as G
#from rdkit import Chem



rules = G.gram.split('\n')

MAX_LEN = 277
DIM = len(rules)
LATENT = 2 #292
EPOCHS = 20
BATCH = 500


def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular autoencoder network')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT,
                        help='Dimensionality of the latent representation.')
    return parser.parse_args()


def cond_sample2(x): # x - (N,MAX_LEN,DIM)
        
    samples = np.array([]).reshape(0,MAX_LEN,DIM)
    for i in range(x.shape[0]):
        a_samp = np.array([]).reshape(1,0,DIM)
        stack = [0]
        count = 0
        while True:
            if not stack:
                break
            if count == MAX_LEN:
                #print('still has some nts... oh well!')
                break
                #print('not done generating...')
                #print('try again!')
                #count = 0
                #stack = [0]
                #a_samp = np.array([]).reshape(1,0,DIM)
            current_nt = stack.pop(0)
            masked = np.exp(x[i,count,:]) * G.masks[current_nt]
            softmax = masked / np.sum(masked, axis=-1)
            choice = np.array([np.argmax(softmax)])
            #choice = np.random.choice(DIM, size=1, p=softmax)
            s = many_one_hot(choice, DIM).reshape(1,1,DIM)
            a_samp = np.concatenate((a_samp, s), axis=1)
            #stack.extend(G.rhs_map[choice])
            stack = G.rhs_map[choice] + stack
            count = count + 1
        if a_samp.shape[1] < MAX_LEN:
            extra = np.repeat(many_one_hot(np.array([DIM-1]), DIM),(MAX_LEN-a_samp.shape[1]),axis=0).reshape(1,(MAX_LEN-a_samp.shape[1]),DIM)
            a_samp = np.concatenate((a_samp,extra), axis=1)
        samples = np.concatenate((samples, a_samp))
    return samples


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
            rule_list.append(G.GCFG.productions()[ix])
        sn_rules.append(prod_to_string(rule_list,''))

    return np.array(sn_rules)


def mask_it(X):
    X2 = X.argmax(axis=-1).flatten()
    ix2 = G.ind_of_ind[X2]
    M2 = G.masks[ix2,:]
    M3 = M2.reshape((5000, MAX_LEN, DIM))
    return M3


# NOT YET USED!!!!!!!!!!
def all_ll(model, filename, XTE, args):

    if os.path.isfile(filename):
        model.load(rules, filename, latent_rep_size = args.latent_dim, max_length=MAX_LEN)
    else:
        raise ValueError("Model file %s doesn't exist" % filename)

    (Z_M, Z_LV) = model.encoderMV.predict(XTE)
    prior = multivariate_normal(np.ones((args.latent_dim,)), np.eye(args.latent_dim))
    ne = XTE.shape[0]
    ll = np.zeros((ne,))
    ALL_STRINGS = np.chararray((5000,100,10), itemsize=1000)

    # For each test point:
    # 100 samples of Z
    # 10 samples of X


    STRINGS = np.chararray((100,10), itemsize=1000)
    CORR = np.zeros((100,10))
    for i in range(ne):
        XTEi = XTE[i].reshape((1,MAX_LEN, DIM))
        XTE_string = get_strings2(XTEi)
        many = np.repeat(XTEi, 100, axis=0)
        Z = model.encoder.predict(many) # encode 1000
        enc = multivariate_normal(Z_M[i], 0.01*np.diag(np.exp(Z_LV[i]/2.0)))
        qZX = enc.pdf(Z)
        pZ = prior.pdf(Z)
        
        lqZX = np.log(qZX)
        lpZ  = np.log(pZ)

        # now get pXZ but use indicator!
        unnorm = model.decoder.predict(Z)

        for jj in range(10): 
            x_cond = cond_sample(unnorm)
            x_string = get_strings2(x_cond) # 100
            STRINGS[:,jj] = x_string
            
        for k in range(100):
            for l in range(10):
                CORR[k,l] = float(XTE_string == STRINGS[k,l])
        
        pXZ = np.mean(CORR,axis=-1)  # 100
        ALL_STRINGS[i] = STRINGS

        log_diff = (lpZ - lqZX)
        like = np.mean(pXZ * np.exp(log_diff))

        #####AA = np.sum(soft_mask * XTEi, axis=-1)
        #####lAA = np.log(AA)
        #####lpXZ = np.sum(lAA, axis=-1)

        #####log_imp = (lpXZ + lpZ) - lqZX
        #####MAX = np.max(log_imp)
        # from: http://blog.smola.org/post/987977550/log-probabilities-semirings-and-floating-point
        #####log_like = np.log(np.sum(np.exp(log_imp - MAX))) + MAX
        #pXZ = np.mean(np.sum(soft_mask * XTEi, axis=-1),axis=-1)
        #pXZ = np.prod(np.sum(soft_mask * XTEi, axis=-1),axis=-1)
        #like = np.mean((CORR * pZ) / qZX)

        ll[i] = np.log(like)

        #ll[i] = log_like
    nll = -ll
    return (nll, ALL_STRINGS)

def importance_ll(model, filename, XTE, args, mask_flag):

    if os.path.isfile(filename):
        model.load(rules, filename, latent_rep_size = args.latent_dim, max_length=MAX_LEN)
    else:
        raise ValueError("Model file %s doesn't exist" % filename)

    (Z_M, Z_LV) = model.encoderMV.predict(XTE)
    prior = multivariate_normal(np.ones((args.latent_dim,)), np.eye(args.latent_dim))
    if mask_flag == 1:
        masks_TE = mask_it(XTE)
    ne = XTE.shape[0]
    ll = np.zeros((ne,))
    for i in range(ne):
        XTEi = XTE[i].reshape((1,MAX_LEN, DIM))
        many = np.repeat(XTEi, 1000, axis=0)
        Z = model.encoder.predict(many)
        enc = multivariate_normal(Z_M[i], 0.01*np.diag(np.exp(Z_LV[i]/2.0)))
        qZX = enc.pdf(Z)
        pZ = prior.pdf(Z)
        
        lqZX = np.log(qZX*1000)
        lpZ  = np.log(pZ)

        unnorm = model.decoder.predict(Z)

        if mask_flag == 1:
            soft = np.exp(unnorm) * masks_TE[i]
            soft_mask = soft / np.sum(soft,axis=-1,keepdims=True)
        else:
            soft = np.exp(unnorm)
            soft_mask = soft / np.sum(soft,axis=-1,keepdims=True)
        
        AA = np.sum(soft_mask * XTEi, axis=-1)
        lAA = np.log(AA)
        lpXZ = np.sum(lAA, axis=-1)

        log_imp = (lpXZ + lpZ) - lqZX
        MAX = np.max(log_imp)
        # from: http://blog.smola.org/post/987977550/log-probabilities-semirings-and-floating-point
        log_like = np.log(np.sum(np.exp(log_imp - MAX))) + MAX
        pdb.set_trace()
        #pXZ = np.mean(np.sum(soft_mask * XTEi, axis=-1),axis=-1)
        #pXZ = np.prod(np.sum(soft_mask * XTEi, axis=-1),axis=-1)
        #like = np.mean((pXZ * pZ) / qZX)
        #ll[i] = np.log(like)
        ll[i] = log_like

    return ll



def cond_sample(x): # x - (N,MAX_LEN,DIM)
        
    samples = np.array([]).reshape(0,MAX_LEN,DIM)
    for i in range(x.shape[0]):
        a_samp = np.array([]).reshape(1,0,DIM)
        stack = [0]
        count = 0
        while True:
            if not stack:
                break
            if count == MAX_LEN:
                print('not done generating...')
                #print('try again!')
                break
                count = 0
                stack = [0]
                a_samp = np.array([]).reshape(1,0,DIM)
            current_nt = stack.pop(0)
            masked = np.exp(x[i,count,:]) * G.masks[current_nt]
            softmax = masked / np.sum(masked, axis=-1)
            choice = np.random.choice(DIM, size=1, p=softmax)
            s = many_one_hot(choice, DIM).reshape(1,1,DIM)
            a_samp = np.concatenate((a_samp, s), axis=1)
            #stack.extend(G.rhs_map[choice])
            stack = G.rhs_map[choice[0]] + stack
            count = count + 1
        if a_samp.shape[1] < MAX_LEN:
            extra = np.repeat(many_one_hot(np.array([DIM-1]), DIM),(MAX_LEN-a_samp.shape[1]),axis=0).reshape(1,(MAX_LEN-a_samp.shape[1]),DIM)
            a_samp = np.concatenate((a_samp,extra), axis=1)
        samples = np.concatenate((samples, a_samp))
    return samples



def full_ll_savestr(model, filename, XTE, args):

    if os.path.isfile(filename):
        model.load(rules, filename, latent_rep_size = args.latent_dim, max_length=MAX_LEN)
    else:
        raise ValueError("Model file %s doesn't exist" % filename)

    #masks_TE = mask_it(XTE)
    ne = XTE.shape[0]
    #OH = np.zeros((1000*10*MAX_LEN,DIM))
    #I = range(1000*10*MAX_LEN)
    ll = np.zeros((ne,))

    ALL_STRINGS = np.chararray((5000,1000), itemsize=1000)
    for i in range(ne):
        print('i=' + str(i) + ' out of 5000')
        XTEi = XTE[i].reshape((1,MAX_LEN, DIM))
        many = np.repeat(XTEi, 10, axis=0)
        Z = model.encoder.predict(many)
        unnorm = model.decoder.predict(Z) # (10,MAX_LEN,DIM)
#        masked = unnorm * masks_TE[i]
#        masked[masked==0] = -9999 # a clear hack
#        masked = masked.reshape((1,10,MAX_LEN,DIM)) 
        samples = np.zeros((1000,MAX_LEN,DIM))

        STR_XTEi = get_strings2(XTEi)
        CORR = np.zeros((1000,))
        STRINGS = np.chararray((1000,), itemsize=1000)
        for j in range(100):
            samples[(j*10):(j+1)*10] = cond_sample(unnorm)
            STRINGS[(j*10):(j+1)*10] = get_strings2(samples[(j*10):(j+1)*10])
         
        for jj in range(1000):
            CORR[jj] = float(STR_XTEi == STRINGS[jj])
            
        ALL_STRINGS[i,:] = STRINGS
        ll[i] = np.mean(CORR)
         

    return (ll,ALL_STRINGS)


def full_ll(model, filename, XTE, args):

    if os.path.isfile(filename):
        model.load(rules, filename, latent_rep_size = args.latent_dim, max_length=MAX_LEN)
    else:
        raise ValueError("Model file %s doesn't exist" % filename)

    #masks_TE = mask_it(XTE)
    ne = XTE.shape[0]
    #OH = np.zeros((1000*10*MAX_LEN,DIM))
    #I = range(1000*10*MAX_LEN)
    ll = np.zeros((ne,))
    for i in range(ne):
        print('i=' + str(i) + ' out of 5000')
        XTEi = XTE[i].reshape((1,MAX_LEN, DIM))
        many = np.repeat(XTEi, 10, axis=0)
        Z = model.encoder.predict(many)
        unnorm = model.decoder.predict(Z) # (10,MAX_LEN,DIM)
#        masked = unnorm * masks_TE[i]
#        masked[masked==0] = -9999 # a clear hack
#        masked = masked.reshape((1,10,MAX_LEN,DIM)) 
        samples = np.zeros((1000,MAX_LEN,DIM))
        for j in range(100):
            samples[(j*10):(j+1)*10] = cond_sample(unnorm)
#        G = np.random.gumbel(size=(1000,10,MAX_LEN,DIM))
#        noise_mask = G + masked

#        samples = noise_mask.argmax(axis=-1)
#        S = samples.reshape((1000*10*MAX_LEN,))
#        OH[I,S] = 1
#        SOH = OH.reshape((10000,MAX_LEN,DIM))
#        combined = np.sum(SOH*XTEi,axis=-1)
        combined = np.sum(samples*XTEi,axis=-1)
        ll[i] = np.mean(np.all(combined,axis=-1))
        ##ll[i] = np.mean(np.sum(SOH*XTEi,axis=-1))
#        OH[:,:] = 0

        print('mean reconstruct=' + str(np.mean(ll[:i+1])))

    return ll





def main():


    h5f = h5py.File('zinc_dataset_sampled.h5', 'r')
    XTR = h5f['XTR'][:]
    XTE = h5f['XTE'][:]
    h5f.close()

    args = get_arguments()
    model = MoleculeVAE()

    if args.latent_dim == 292:
        filename = 'results/zinc_vae_L292_50?.hdf5'
    else:
        filename = 'results/zinc_vae_L' + str(args.latent_dim) + '.hdf5'

    print(filename)


    save_full = 'results/acc_full_zinc_L' + str(args.latent_dim) + '.p'

#!    if not os.path.exists(save_full):
    print('running full_ll')
    acc_full = full_ll(model, filename, XTE, args)
    print('acc full=', str(np.mean(acc_full)))
#!        pickle.dump({'acc_full': acc_full}, open(save_full, 'wb')) #open('results/nll_str_equation_L' + str(args.latent_dim) + '.p', 'wb'))
#!    else:
#!        print('already ran full_ll')


    ###save_full_smiles = 'results/acc_full_smiles_zinc_L' + str(args.latent_dim) + '.p'

    ###if not os.path.exists(save_full_smiles):
    ###    print('running full_ll smiles')
    ###    (acc_full_smiles, ALL_STRINGS) = full_ll_savestr(model, filename, XTE, args)
    ###    print('acc full smiles=', str(np.mean(acc_full_smiles)))
    ###    pickle.dump({'acc_full_smiles': acc_full_smiles, 'ALL_STRINGS': ALL_STRINGS}, open(save_full_smiles, 'wb')) #open('results/nll_str_equation_L' + str(args.latent_dim) + '.p', 'wb'))
    ###else:
    ###    print('already ran full_ll smiles')



    ##save_all_ll =  'results/nll_all_zinc_L' + str(args.latent_dim)

    ##if not os.path.exists(save_all_ll + '.p'):
    ##    print('running importance_all_ll')
    ##    (nll, ALL_STRINGS) = all_ll(model, filename, XTE, args)
    ##    print('nll=', str(np.sum(nll)))
    ##    pickle.dump({'all_nll': nll, 'ALL_STRINGS': ALL_STRINGS}, open(save_all_ll + '.p', 'wb'))
    ##else:
    ##    print('already ran importance_all_ll')



    save_imp = 'results/nll_prod_zinc_L' + str(args.latent_dim)
    ###
    ###if not os.path.exists(save_imp + '.p'):
    ###    print('running importance_ll')
    ###    nll = -importance_ll(model, filename, XTE, args, 0)
    ###    print('nll=', str(np.sum(nll)))
    ###    pickle.dump({'nll': nll}, open(save_imp + '.p', 'wb'))
    ###else:
    ###    print('already ran importance_ll')


    #if not os.path.exists(save_imp + '_mask.p'):
    print('running importance_ll mask')
    nll = -importance_ll(model, filename, XTE, args, 1)
    print('nll mask=', str(np.sum(nll)))
    pickle.dump({'nll': nll}, open(save_imp + '_mask.p' , 'wb'))
    #else:
    print('already ran importance_ll mask')







if __name__ == '__main__':
    main()

