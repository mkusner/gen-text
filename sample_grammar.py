from __future__ import print_function

import argparse
import os
import h5py
import numpy as np
import sys
import nltk
import six

from molecules.model_gr import MoleculeVAE
from molecules.utils import one_hot_array, one_hot_index, from_one_hot_array, \
    decode_smiles_from_indexes, load_dataset

#from pylab import figure, axes, scatter, title, show

#from rdkit import Chem
#from rdkit.Chem import Draw

import pdb

#LATENT_DIM = 292
#TARGET = 'autoencoder'

p = """S -> S '+' S
S -> S '*' S
S -> S '/' S
S -> '(' S ')'
S -> 'sin(' S ')'
S -> 'exp(' S ')'
S -> 'x'
S -> '1'
S -> '2'
S -> '3'
S -> ' '"""

rules = p.split('\n')
gr = nltk.CFG.fromstring(p)


MAX_LEN = 7
DIM = len(rules)
LATENT = 2
EPOCHS = 10
BATCH = 600

####def get_arguments():
####    parser = argparse.ArgumentParser(description='Molecular autoencoder network')
####    parser.add_argument('data', type=str, help='File of latent representation tensors for decoding.')
####    parser.add_argument('model', type=str, help='Trained Keras model to use.')
####    parser.add_argument('--save_h5', type=str, help='Name of a file to write HDF5 output to.')
####    parser.add_argument('--target', type=str, default=TARGET,
####                        help='What model to sample from: autoencoder, encoder, decoder.')
####    parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT_DIM,
####                        help='Dimensionality of the latent representation.')
####    return parser.parse_args()

def read_latent_data(filename):
    h5f = h5py.File(filename, 'r')
    data = h5f['latent_vectors'][:]
    charset =  h5f['charset'][:]
    h5f.close()
    return (data, charset)

def autoencoder(args, model):
    latent_dim = args.latent_dim
    data, charset = load_dataset(args.data, split = False)

    if os.path.isfile(args.model):
        model.load(charset, args.model, latent_rep_size = latent_dim)
    else:
        raise ValueError("Model file %s doesn't exist" % args.model)

    sampled = model.autoencoder.predict(data[0].reshape(1, 120, len(charset))).argmax(axis=2)[0]
    mol = decode_smiles_from_indexes(map(from_one_hot_array, data[0]), charset)
    sampled = decode_smiles_from_indexes(sampled, charset)
    print(mol)
    print(sampled)

def decoder(args, model):
    latent_dim = args.latent_dim
    data, charset = read_latent_data(args.data)

    if os.path.isfile(args.model):
        model.load(charset, args.model, latent_rep_size = latent_dim)
    else:
        raise ValueError("Model file %s doesn't exist" % args.model)

    sampled = model.decoder.predict(data[0].reshape(1, latent_dim)).argmax(axis=2)[0]
    sampled = decode_smiles_from_indexes(sampled, charset)
    print(sampled)


def encoder(args, model):
    latent_dim = args.latent_dim
    data, charset = load_dataset(args.data, split = False)

    if os.path.isfile(args.model):
        model.load(charset, args.model, latent_rep_size = latent_dim)
    else:
        raise ValueError("Model file %s doesn't exist" % args.model)

    x_latent = model.encoder.predict(data)
    if args.save_h5:
        h5f = h5py.File(args.save_h5, 'w')
        h5f.create_dataset('charset', data = charset)
        h5f.create_dataset('latent_vectors', data = x_latent)
        h5f.close()
    else:
        np.savetxt(sys.stdout, x_latent, delimiter = '\t')

def decode_interp(model, filename):
    if os.path.isfile(filename):
        model.load(rules, filename, latent_rep_size = LATENT)
    else:
        raise ValueError("Model file %s doesn't exist" % filename)
    interp02 = np.linspace(0,3,10)
    interpn22= np.linspace(0,-3,10)

    P = np.zeros((10,2))
    # pos x-axis
    P[:,0] = interp02
    P[:,1] = 0 #interp02
    sampled1 = model.decoder.predict(P).argmax(axis=2) # (10,7)
    
    # neg x-axis
    P[:,0] = interpn22
    P[:,1] = 0
    sampled2 = model.decoder.predict(P).argmax(axis=2) # (10,7)

    # pos y-axis
    P[:,0] = 0
    P[:,1] = interp02
    sampled3 = model.decoder.predict(P).argmax(axis=2) # (10,7)

    # neg y-axis
    P[:,0] = 0
    P[:,1] = interpn22
    sampled4 = model.decoder.predict(P).argmax(axis=2) # (10,7)

    # between neg y-axis and pos x-axis
    P[:,0] = interp02
    P[:,1] = interpn22
    sampled5 = model.decoder.predict(P).argmax(axis=2) # (10,7)
    return (sampled1,sampled2,sampled3,sampled4,sampled5)



def prod_to_string(P,string):
    if len(P) == 0:
        return string
    tup = P[0].rhs()
    for item in tup:
        if len(P) == 0:
            return string
        if isinstance(item,six.string_types):
            string = string + ' ' + item
        else:
            P.pop(0)
            string = prod_to_string(P, string)
    return string


def get_strings(sn):

    sn_rules = []
    for s in sn:
        rule_list = []
        for r in s:
            rule_list.append(gr.productions()[r])
        sn_rules.append(prod_to_string(rule_list,''))

    return sn_rules

def main():
    #args = get_arguments()
    model = MoleculeVAE()
    filename = 'eq_vae_h50_c123.hdf5'
    #pdb.set_trace()
    (s1,s2,s3,s4,s5) = decode_interp(model, filename)
    #pdb.set_trace()
    the1 = get_strings(s1)
    the2 = get_strings(s2)
    the3 = get_strings(s3)
    the4 = get_strings(s4)
    the5 = get_strings(s5)


    pdb.set_trace()
    ###if args.target == 'autoencoder':
    ###    autoencoder(args, model)
    ###elif args.target == 'encoder':
    ###    encoder(args, model)
    ###elif args.target == 'decoder':
    ###    decoder(args, model)

if __name__ == '__main__':
    main()
