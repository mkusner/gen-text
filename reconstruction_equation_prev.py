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
import the_grammar as G
#from rdkit import Chem

from sklearn.metrics.pairwise import euclidean_distances

from keras import backend as K

rules = G.gram.split('\n')

MAX_LEN = 15
DIM = len(rules)
LATENT = 2 #292
EPOCHS = 10
BATCH = 500
BATCH = 1

def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular autoencoder network')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT,
                        help='Dimensionality of the latent representation.')
    return parser.parse_args()



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


def compare_chars(X, X_R):
    n = X.shape[0]
    right = 0
    tot = 0
    full_right = 0
    full_tot = 0
    for i in range(n):
        ml = len(X[i])
        ml_r = len(X_R[i])
        ml_min = np.min([ml,ml_r])
        right = right + np.sum(np.array(list(X[i][0:ml_min])) == np.array(list(X_R[i][0:ml_min])))
        if (ml == ml_r) and (np.all(X[i][0:ml_min] == X_R[i][0:ml_min])):
            full_right = full_right + 1
        
        tot = tot + ml
        full_tot = full_tot + 1
    
    return (right,tot,full_right,full_tot)


def reconstruct(model, filename, XTR, XTE, args):
    if os.path.isfile(filename):
        model.load(rules, filename, latent_rep_size = args.latent_dim, max_length=MAX_LEN)
    else:
        raise ValueError("Model file %s doesn't exist" % filename)
    n = XTR.shape[0]
    
    #rec_file = 'results/recon_zinc_XTR_L' + str(args.latent_dim) + '_50?.hdf5'
    rec_file = 'results/recon_zinc_XTR_L' + str(args.latent_dim) + '_50?div3.hdf5'
    if os.path.exists(rec_file):
        print('preloaded XTR!')
        h5f = h5py.File(rec_file, 'r')
        XTR_strings = h5f['XTR_strings'][:]
        XTR_R_strings = h5f['XTR_R_strings'][:]
        h5f.close()
    else:
        print('need to compute XTR...')
        #ZTR = np.zeros((n,LATENT))
        #XTR_R = np.zeros((n,XTR.shape[1],XTR.shape[2]))
        XTR_strings = np.chararray((n,), itemsize=1000)
        XTR_R_strings = np.chararray((n,), itemsize=1000)

        times = n/1000
        for t in range(times):
            batch = XTR[(t*1000):((t+1)*1000)]
            XTR_strings[(t*1000):((t+1)*1000)] = get_strings2(batch)
            z_batch = model.encoder.predict(batch)
            #ZTR[(t*1000):((t+1)*1000)] = z_batch
            x_r_batch = model.decoder.predict(z_batch)
            x_cond = cond_sample2(x_r_batch)
            #XTR_R[(t*1000):((t+1)*1000)] = x_cond
            XTR_R_strings[(t*1000):((t+1)*1000)] = get_strings2(x_cond)
        #XTR_strings = XTR_strings.astype(str)
        #XTR_R_strings = XTR_R_strings.astype(str)
        h5f = h5py.File(rec_file, 'w')
        h5f.create_dataset('XTR_strings', data = XTR_strings)
        h5f.create_dataset('XTR_R_strings', data = XTR_R_strings)
        h5f.close()

    (right_xtr,tot_xtr,full_right_xtr,full_tot_xtr) = compare_chars(XTR_strings, XTR_R_strings)

    print('XTR by char acc=' + str(right_xtr/(1.0*tot_xtr)))
    print('XTR full acc=' + str(full_right_xtr/(1.0*full_tot_xtr)))

    acc_by_char_xtr = (right_xtr/(1.0*tot_xtr))
    acc_full_xtr = (full_right_xtr/(1.0*full_tot_xtr))

    ne = XTE.shape[0]
    #ZTE = np.zeros((ne,LATENT))
    #XTE_R = np.zeros(XTE.shape)
    #rec_file = 'results/recon_zinc_XTE_L' + str(args.latent_dim) + '_50?.hdf5'
    rec_file = 'results/recon_zinc_XTE_L' + str(args.latent_dim) + '_50?div3.hdf5'
    if os.path.exists(rec_file):
        print('preloaded XTE!')
        h5f = h5py.File(rec_file, 'r')
        XTE_strings = h5f['XTE_strings'][:]
        XTE_R_strings = h5f['XTE_R_strings'][:]
        h5f.close()
    else:
        print('need to compute XTE...')
        XTE_strings = np.chararray((ne,), itemsize=1000)
        XTE_R_strings = np.chararray((ne,), itemsize=1000)

        times = ne/1000
        for t in range(times):
            batch = XTE[(t*1000):((t+1)*1000)]
            XTE_strings[(t*1000):((t+1)*1000)] = get_strings2(batch)
            z_batch = model.encoder.predict(batch)
            #ZTR[(t*1000):((t+1)*1000)] = z_batch
            xe_r_batch = model.decoder.predict(z_batch)
            xe_cond = cond_sample2(xe_r_batch)
            #XTR_R[(t*1000):((t+1)*1000)] = x_cond
            XTE_R_strings[(t*1000):((t+1)*1000)] = get_strings2(xe_cond)
        h5f = h5py.File(rec_file, 'w')
        h5f.create_dataset('XTE_strings', data = XTE_strings)
        h5f.create_dataset('XTE_R_strings', data = XTE_R_strings)
        h5f.close()


    (right_xte,tot_xte,full_right_xte,full_tot_xte) = compare_chars(XTE_strings, XTE_R_strings)

    print('XTE by char acc=' + str(right_xte/(1.0*tot_xte)))
    print('XTE full acc=' + str(full_right_xte/(1.0*full_tot_xte)))

    acc_by_char_xte = (right_xte/(1.0*tot_xte))
    acc_full_xte = (full_right_xte/(1.0*full_tot_xte))

    return (acc_by_char_xtr, acc_full_xtr, acc_by_char_xte, acc_full_xte, tot_xtr, full_tot_xtr, tot_xte, full_tot_xte)

    
##def four_point_interp(model, filename, points):
##
##    # points (4,latent)
##    from1to2 = (points[2] - points[0]) / 5.0
##    from4to3 = (points[3] - points[4]) / 5.0
##
##    for i in range(6):
##        for j in range(6):


def hard_sigmoid(x):
    x = 0.2*x + 0.5
    return np.clip(x, 0.0, 1.0)

def numpy_decode(model, latent)
    # model - keras model
    # latent - numpy output from encoder
    decoder = model.decoder
    decoder.layers = decoder.layers[:-1]
    inp = model.decoder.layers[0].input
    next_to_last = vae.decoder.layers[-2].output
    get_output_from_input = K.function([inp],[next_to_last])
    output = get_output_from_input([latent])[0]     # (batch, max_len, D)

    if on_gpu == 1:
    else:
    W_z = K.eval(model.decoder.layers[-1].W_z)
    W_h = K.eval(model.decoder.layers[-1].W_h)
    W_r = K.eval(model.decoder.layers[-1].W_r)
    U_z = K.eval(model.decoder.layers[-1].U_z)
    b_z = K.eval(model.decoder.layers[-1].W_z)
    b_h = K.eval(model.decoder.layers[-1].b_h)
    b_r = K.eval(model.decoder.layers[-1].b_r)
    U_z = K.eval(model.decoder.layers[-1].U_z)


# DO THIS!
def two_D_plane(model, filename, point, args, mult):
    if os.path.isfile(filename):
        model.load(rules, BATCH, filename, latent_rep_size = args.latent_dim, max_length=MAX_LEN)
    else:
        raise ValueError("Model file %s doesn't exist" % filename)
    pdb.set_trace()
    #z_strings = get_strings2(point)  #point.reshape(1, MAX_LEN, DIM))
    #n = z_strings.shape[0]
    #LENS = np.zeros((n,))
    #for i in range(n):
    #    LENS[i] = len(z_strings[i])
    #    print(z_strings[i])

    #pdb.set_trace()
    z_string = get_strings2(point.reshape(1, MAX_LEN, DIM))
    z_point = model.encoder.predict(point.reshape(1, MAX_LEN, DIM))
    
    np.random.seed(1)
    vec1 = np.random.normal(0,1,args.latent_dim)
    vec2 = np.random.normal(0,1,args.latent_dim)
    vec2[-1] = -np.sum(vec1[0:-1]*vec2[0:-1])/vec1[-1]

    ##vec2 = np.cross(vec1.reshape((args.latent_dim,1)), vec2.reshape((args.latent_dim,1)))
    vec1 = vec1 / (np.linalg.norm(vec1)*mult)
    vec2 = vec2 / (np.linalg.norm(vec2)*mult)

    to_add = range(-6,0)
    to_add.extend(range(0,7))
    results = np.chararray((len(to_add),len(to_add),1000), itemsize='1000')

    count_i = 0
    PLACEHOLD = np.zeros((1,MAX_LEN,DIM))
    for i in to_add:
        displace1 = vec1*i
        count_j = 0
        for j in to_add:
            displace2 = vec2*j
            new = z_point + displace1 + displace2
            x_new = model.decoder.predict([new,PLACEHOLD])
            pdb.set_trace()
            for k in range(1000):
                the_string = get_strings2(x_new)
                #x_cond = cond_sample(x_new)
                #the_string = get_strings2(x_cond)
            #x_cond = cond_sample2(x_new)
                results[count_i,count_j,k] = the_string[0]
            count_j = count_j + 1
        count_i = count_i + 1
            # DECODE HERE
    results[6,6,0] = z_string[0]
    return results

def find_good_point(model, filename, XTR, args):
    if os.path.isfile(filename):
        model.load(rules, BATCH, filename, latent_rep_size = args.latent_dim, max_length=MAX_LEN)
    else:
        raise ValueError("Model file %s doesn't exist" % filename)
    n = XTR.shape[0]
    Zs = np.zeros((n,args.latent_dim))
    for i in range(n/100):
        Zs[(i*100):((i+1)*100)] = model.encoder.predict(XTR[(i*100):((i+1)*100)])
    EZ = euclidean_distances(Zs, Zs)
    EO = euclidean_distances(Zs, np.zeros((1,args.latent_dim)))
    print('EO min=' + str(np.argmin(EO)))
    EZS = np.sum(EZ,axis=-1)
    print('EZS min=' + str(np.argmin(EZS)))
    pdb.set_trace()
    # 3020 - for 292
    # 56
    #EO min=6427
    #EZS min=6528



def main():


    h5f = h5py.File('eq2_15_dataset.h5', 'r')
    data = h5f['data'][:]
    h5f.close()
    np.random.seed(0)
    N = data.shape[0]
    IND = range(N)
    np.random.shuffle(IND)
    point = data[IND[0]]

    #point = data[108055]

    args = get_arguments()
    K.set_learning_phase(0)
    model = MoleculeVAE()
    filename = 'results/eq_prev_vae_h50_c123_cond_L' + str(args.latent_dim) + '.hdf5'
    print(filename)
    #find_good_point(model, filename, np.concatenate((XTR, XTE)), args)
##    point = X[6528]
##    point = X[6427]

    np.random.seed(1)
    #N = XTR.shape[0]
    #IND = range(N)
    #np.random.shuffle(IND)
    #point = XTR[IND[0]]
    #point = XTR[IND[0:100]]
    #point = XTR
    #point = XTR[4994]
    #smiles = 'Cc1cc(F)cc(C(CO)CC(C)(C)C)n1'
    mults = [3,4,5,6]
    mults = [9,10]
    mults = [11,12,13,14,15]
    mults = [2,4,6,8,10]
    for mm in mults:
        print('on mult=' + str(mm))
        results = two_D_plane(model, filename, point, args, mm)
    #filename = 'results/zinc_vae.hdf5'
# add random seed here before two_D_plane
    
    #pickle.dump({'results': results}, open('results/reconstruction_zinc_L' + str(args.latent_dim) + '_50?div3_x.p', 'wb'))
    #pickle.dump({'results': results}, open('results/reconstruction_zinc_L' + str(args.latent_dim) + '_E20.p', 'wb'))
##        pickle.dump({'results': results}, open('results/reconstruction_zinc_L' + str(args.latent_dim) + '_6528_M' + str(mm) + '.p', 'wb'))
        pickle.dump({'results': results}, open('results/reconstruction_eq_prev_L' + str(args.latent_dim) + '_M' + str(mm) + '.p', 'wb'))
    pdb.set_trace()

    (acc_by_char_xtr, acc_full_xtr, acc_by_char_xte, acc_full_xte, tot_xtr,full_tot_xtr,tot_xte,full_tot_xte) = reconstruct(model, filename, XTR, XTE, args)

    pickle.dump({'results': results, 'acc_by_char_xtr': acc_by_char_xtr, 'acc_full_xtr': acc_full_xtr, 'acc_by_char_xte': acc_by_char_xte, 'acc_full_xte': acc_full_xte, 'tot_xtr': tot_xtr, 'full_tot_xtr': full_tot_xtr, 'tot_xte': tot_xte,'full_tot_xte': full_tot_xte}, open('results/reconstruction_zinc_L' + str(args.latent_dim) + '_50?div3.p', 'wb'))

    








    ##ALL = decode_interp(model, filename) # 100, MAX_LEN, DIM
    ##strs = np.array([]).reshape(0,ALL.shape[1])
    ##for i in range(ALL.shape[0]):
    ##    sn_rules = get_strings2(ALL[i])
    ##    strs = np.concatenate((strs,sn_rules.reshape(1,ALL.shape[1])))
    ##    for j in range(ALL.shape[1]):
    ##        print('i='+str(i)+', j='+str(j))
    ##        m = Chem.MolFromSmiles(strs[i,j])

    ##pdb.set_trace()


if __name__ == '__main__':
    main()
