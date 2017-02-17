from __future__ import print_function

import argparse
import os
import h5py
import numpy as np

#from molecules.model_gr_prev import MoleculeVAE
from molecules.model_gr import MoleculeVAE
from molecules.utils import one_hot_array, one_hot_index, from_one_hot_array, \
    decode_smiles_from_indexes, load_dataset
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

import h5py
import the_grammar as G
###NUM_EPOCHS = 1
###BATCH_SIZE = 600
###LATENT_DIM = 292


MAX_LEN = 15
LATENT = 2
EPOCHS = 100
BATCH = 600




###p = """S -> S '+' S
###S -> S '*' S
###S -> S '/' S
###S -> '(' S ')'
###S -> 'sin(' S ')'
###S -> 'exp(' S ')'
###S -> 'x'
###S -> '1'
###S -> '2'
###S -> '3'
###"""


rules = G.gram.split('\n')


DIM = len(rules)


def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular autoencoder network')
    parser.add_argument('--epochs', type=int, metavar='N', default=EPOCHS,
                        help='Number of epochs to run during training.')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT,
                        help='Dimensionality of the latent representation.')
    return parser.parse_args()

def main():

    h5f = h5py.File('eq2_15_dataset.h5', 'r')
    data = h5f['data'][:]
    h5f.close()

    args = get_arguments()

    #model_save = '/Users/matthewkusner/Dropbox/gen-text/eq_vae_h50_c123.hdf5'
    #####model_save = '/Users/matthewkusner/Dropbox/gen-text/eq_vae_h100_c123_cond20.hdf5'
    #model_save = '/Users/matthewkusner/Dropbox/gen-text/eq_vae_h50_c113.hdf5'
    #args = get_arguments()
    #data_train, data_test, charset = load_dataset(args.data)

    params = {'hidden': 200, 'dense': 200, 'conv1': 2, 'conv2': 3, 'conv3': 4}

    model_save = '/Users/matthewkusner/Dropbox/gen-text/results/eq_prev_train_vae_h50_c123_cond_L' + str(args.latent_dim) + '.hdf5'
    model_save = '/Users/matthewkusner/Dropbox/gen-text/results/eq_vae_grammar_h' + str(params['hidden']) + '_c123_L' + str(args.latent_dim) + '_E' + str(args.epochs) + '.hdf5'

    model = MoleculeVAE()
    if os.path.isfile(model_save):
        model.load(rules, model_save, latent_rep_size = args.latent_dim, hypers = params)
    else:
        model.create(rules, max_length=MAX_LEN, latent_rep_size = args.latent_dim, hypers = params)

    checkpointer = ModelCheckpoint(filepath = model_save,
                                   verbose = 1,
                                   save_best_only = True)

    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                  factor = 0.2,
                                  patience = 1,
                                  min_lr = 0.0001)

    model.autoencoder.fit(
        data,
        data,
        shuffle = True,
        nb_epoch = args.epochs,
        batch_size = BATCH,
        callbacks = [checkpointer, reduce_lr],
        validation_split = 0.1
    )

if __name__ == '__main__':
    main()
