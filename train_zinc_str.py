from __future__ import print_function

import argparse
import os
import h5py
import numpy as np

from molecules.model import MoleculeVAE
from molecules.utils import one_hot_array, one_hot_index, from_one_hot_array, \
    decode_smiles_from_indexes, load_dataset
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

import h5py
##import zinc_grammar as G
import pdb
###NUM_EPOCHS = 1
###BATCH_SIZE = 600
###LATENT_DIM = 292





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


#rules = G.gram.split('\n')

charset = ['C', '(', ')', 'c', '1', '2', 'o', '=', 'O', 'N', '3', 'F', '[', '@', 'H', ']', 'n', '-', '#', 'S', 'l', '+', 's', 'B', 'r', '/', '4', '\\', '5', '6', '7', 'I', 'P', '8', ' ']

MAX_LEN = 120
DIM = len(charset)
LATENT = 292
EPOCHS = 100
BATCH = 500 #600


def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular autoencoder network')
    parser.add_argument('--load_model', type=str, metavar='N', default="")
#####    parser.add_argument('data', type=str, help='The HDF5 file containing preprocessed data.')
#####    parser.add_argument('model', type=str,
#####                        help='Where to save the trained model. If this file exists, it will be opened and resumed.')
    parser.add_argument('--epochs', type=int, metavar='N', default=EPOCHS,
                        help='Number of epochs to run during training.')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT,
                        help='Dimensionality of the latent representation.')
#####    parser.add_argument('--batch_size', type=int, metavar='N', default=BATCH_SIZE,
#####                        help='Number of samples to process per minibatch during training.')
    return parser.parse_args()


def main():

    h5f = h5py.File('zinc_str_dataset.h5', 'r')
    data = h5f['data'][:]
    h5f.close()

    np.random.seed(0)
    N = data.shape[0]
    IND = range(N)
    np.random.shuffle(IND)
    ##XTE = data[0:5000]
    ##XTR = data[5000:N]

    #model_save = '/Users/matthewkusner/Dropbox/gen-text/eq_vae_h50_c123.hdf5'
    XTE = data[0:50000]
    XTR = data[50000:N]
    print(XTE.shape)
    #####model_save = '/Users/matthewkusner/Dropbox/gen-text/eq_vae_h100_c123_cond20.hdf5'
    #model_save = '/Users/matthewkusner/Dropbox/gen-text/eq_vae_h50_c113.hdf5'
    args = get_arguments()
    print('L='  + str(args.latent_dim) + ' E=' + str(args.epochs))
    #model_save = 'results/zinc_str_vae_L' + str(args.latent_dim) + '_E' + str(args.epochs) + '_times2.hdf5'
    model_save = 'results/zinc_str_vae_L' + str(args.latent_dim) + '_E' + str(args.epochs) + '.hdf5'
    model_save = 'results/zinc_str_vae_L' + str(args.latent_dim) + '_E{epoch:02d}_BEST.hdf5'
    model_save = 'results/zinc_str_vae_L' + str(args.latent_dim) + '_E' + str(args.epochs) + '_BEST_50K.hdf5'
    print(model_save)
    #model_save = 'results/zinc_str_vae_L292_50_tot.hdf5'
    #data_train, data_test, charset = load_dataset(args.data)
    model = MoleculeVAE()
    print(args.load_model)
    if os.path.isfile(args.load_model):
        print('loading model')
        model.load(charset, args.load_model, latent_rep_size = args.latent_dim, max_length=MAX_LEN)
    else:
        print('making new model')
        model.create(charset, max_length=MAX_LEN, latent_rep_size = args.latent_dim)

    ##checkpointer = ModelCheckpoint(filepath = model_save,
    ##                               verbose = 1) #,
#                                   save_best_only = True)


    checkpointer = ModelCheckpoint(filepath = model_save,
                                   verbose = 1,
                                   save_best_only = True)
    # uncomment for 2D training
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                  factor = 0.2,
                                  patience = 3,
                                  min_lr = 0.0001)

    ##model.autoencoder.fit(
    ##    XTR,
    ##    XTR,
    ##    shuffle = True,
    ##    nb_epoch = args.epochs,
    ##    batch_size = BATCH,
    ##    callbacks = [checkpointer] #, reduce_lr],
    ##    #validation_data = (data_test, data_test)
    ##)


    model.autoencoder.fit(
        XTR,
        XTR,
        shuffle = True,
        nb_epoch = args.epochs,
        batch_size = BATCH,
        callbacks = [checkpointer, reduce_lr],
        validation_data = (XTE, XTE)
    )

if __name__ == '__main__':
    main()
