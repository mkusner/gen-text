from __future__ import print_function

import argparse
import os
import h5py
import numpy as np

from molecules.model_zinc import MoleculeVAE
#from molecules.utils import one_hot_array, one_hot_index, from_one_hot_array, \
#    decode_smiles_from_indexes, load_dataset
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

import h5py
import zinc_grammar as G
import pdb


rules = G.gram.split('\n')


MAX_LEN = 277
DIM = len(rules)
LATENT = 292
EPOCHS = 20
BATCH = 500 #600



def get_arguments():
    parser = argparse.ArgumentParser(description='Molecular autoencoder network')
    parser.add_argument('--epochs', type=int, metavar='N', default=EPOCHS,
                        help='Number of epochs to run during training.')
    parser.add_argument('--latent_dim', type=int, metavar='N', default=LATENT,
                        help='Dimensionality of the latent representation.')
    return parser.parse_args()


def main():
    # 0. load dataset
    h5f = h5py.File('zinc_dataset.h5', 'r')
    data = h5f['data'][:]
    h5f.close()
    
    # 1. split into train/test
    np.random.seed(0)
    N = data.shape[0]
    IND = range(N)
    np.random.shuffle(IND)
    XTE = data[0:5000]
    XTR = data[5000:N]

    # 2. get any arguments and define save file, then create the VAE model
    args = get_arguments()
    model_save = 'results/zinc_vae_L' + str(args.latent_dim) + '.hdf5'
    model = MoleculeVAE()
    if os.path.isfile(model_save):
        print('loading!')
        model.load(rules, model_save, latent_rep_size = args.latent_dim, max_length=MAX_LEN)
    else:
        model.create(rules, max_length=MAX_LEN, latent_rep_size = args.latent_dim)


    checkpointer = ModelCheckpoint(filepath = model_save,
                                   verbose = 1) #,
    # 3. fit the vae
    model.autoencoder.fit(
        XTR,
        XTR,
        shuffle = True,
        nb_epoch = args.epochs,
        batch_size = BATCH,
        callbacks = [checkpointer]
    )

if __name__ == '__main__':
    main()
