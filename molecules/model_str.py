import copy
from keras import backend as K
from keras import objectives
from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.layers.core import Dense, Activation, Flatten, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import GRU
from keras.layers.convolutional import Convolution1D
from keras.layers.normalization import BatchNormalization
#from terminalgru import TerminalGRU
#from recurrentshop import *
#import tensorflow as tf
#import the_grammar as G
#import pdb

#masks_K      = K.variable(G.masks)
#ind_of_ind_K = K.variable(G.ind_of_ind)

MAX_LEN = 19
DIM = 15 #len(rules)
#DIM = G.D


class MoleculeVAE():

    autoencoder = None
    
    def create(self,
               charset,
               max_length = MAX_LEN,
               latent_rep_size = 2,
               hypers = {'hidden': 200, 'dense': 200, 'conv1': 2, 'conv2': 3, 'conv3': 4},
               weights_file = None):
        charset_length = len(charset)

        self.hypers = hypers
        
        x = Input(shape=(max_length, charset_length))
        _, z = self._buildEncoder(x, latent_rep_size, max_length)
        self.encoder = Model(x, z)

        encoded_input = Input(shape=(latent_rep_size,))
        self.decoder = Model(
            encoded_input,
            self._buildDecoder(
                encoded_input,
                latent_rep_size,
                max_length,
                charset_length
            )
        )

        x1 = Input(shape=(max_length, charset_length))
        vae_loss, z1 = self._buildEncoder(x1, latent_rep_size, max_length)
        self.autoencoder = Model(
            x1,
            self._buildDecoder(
                z1,
                latent_rep_size,
                max_length,
                charset_length
            )
        )

        x2 = Input(shape=(max_length, charset_length))
        (z_m, z_l_v) = self._encoderMeanVar(x2, latent_rep_size, max_length)
        self.encoderMV = Model(input=x2, output=[z_m, z_l_v])

        if weights_file:
            self.autoencoder.load_weights(weights_file)
            self.encoder.load_weights(weights_file, by_name = True)
            self.decoder.load_weights(weights_file, by_name = True)
            self.encoderMV.load_weights(weights_file, by_name = True)

        self.autoencoder.compile(optimizer = 'Adam',
                                 loss = vae_loss,
                                 metrics = ['accuracy'])
    def _encoderMeanVar(self, x, latent_rep_size, max_length, epsilon_std = 0.01):

        h = Convolution1D(self.hypers['conv1'], self.hypers['conv1'], activation = 'relu', name='conv_1')(x) # used to be 2, 1
        h = BatchNormalization(name='batch_1')(h)
        h = Convolution1D(self.hypers['conv2'], self.hypers['conv2'], activation = 'relu', name='conv_2')(h) # used to be 2, 1
        h = BatchNormalization(name='batch_2')(h)
        h = Convolution1D(self.hypers['conv3'], self.hypers['conv3'], activation = 'relu', name='conv_3')(h) # used to be 3
        h = BatchNormalization(name='batch_3')(h)

        h = Flatten(name='flatten_1')(h)
        h = Dense(self.hypers['dense'], activation = 'relu', name='dense_1')(h) # used to be 100, 30, 50, 200

        z_mean = Dense(latent_rep_size, name='z_mean', activation = 'linear')(h)
        z_log_var = Dense(latent_rep_size, name='z_log_var', activation = 'linear')(h)

        return (z_mean, z_log_var) 




    def _buildEncoder(self, x, latent_rep_size, max_length, epsilon_std = 0.01):

        h = Convolution1D(self.hypers['conv1'], self.hypers['conv1'], activation = 'relu', name='conv_1')(x) # used to be 2, 1
        h = BatchNormalization(name='batch_1')(h)
        h = Convolution1D(self.hypers['conv2'], self.hypers['conv2'], activation = 'relu', name='conv_2')(h) # used to be 2, 1
        h = BatchNormalization(name='batch_2')(h)
        h = Convolution1D(self.hypers['conv3'], self.hypers['conv3'], activation = 'relu', name='conv_3')(h) # used to be 3
        h = BatchNormalization(name='batch_3')(h)

        #h = Convolution1D(self.hypers['conv1'], self.hypers['conv1'], activation = 'relu', name='conv_1')(x) # used to be 2, 1
        #h = Convolution1D(self.hypers['conv2'], self.hypers['conv2'], activation = 'relu', name='conv_2')(h) # used to be 2, 1
        #h = Convolution1D(self.hypers['conv3'], self.hypers['conv3'], activation = 'relu', name='conv_3')(h) # used to be 3
        h = Flatten(name='flatten_1')(h)
        h = Dense(self.hypers['dense'], activation = 'relu', name='dense_1')(h) # used to be 100, 30, 50, 200

        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]
            epsilon = K.random_normal(shape=(batch_size, latent_rep_size), mean=0., std = epsilon_std)
            return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

        z_mean = Dense(latent_rep_size, name='z_mean', activation = 'linear')(h)
        z_log_var = Dense(latent_rep_size, name='z_log_var', activation = 'linear')(h)

        def vae_loss(x, x_decoded_mean):
            x = K.flatten(x)
            x_decoded_mean = K.flatten(x_decoded_mean)
            xent_loss = max_length * objectives.binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis = -1)
            return xent_loss + kl_loss

        return (vae_loss, Lambda(sampling, output_shape=(latent_rep_size,), name='lambda')([z_mean, z_log_var]))

    def _buildDecoder(self, z, latent_rep_size, max_length, charset_length):
        h = BatchNormalization(name='batch_4')(z)
        h = Dense(latent_rep_size, name='latent_input', activation = 'relu')(h)
        #h = Dense(50, name='latent_input', activation = 'relu')(h)
        h = RepeatVector(max_length, name='repeat_vector')(h)
        h = GRU(self.hypers['hidden'], return_sequences = True, name='gru_1')(h)
        h = GRU(self.hypers['hidden'], return_sequences = True, name='gru_2')(h)
        h = GRU(self.hypers['hidden'], return_sequences = True, name='gru_3')(h) # all above used to be 100, 30, 50, 200
        return TimeDistributed(Dense(charset_length, activation='softmax'), name='decoded_mean')(h)
        #h = SpecialLayer(charset_length, activation='softmax'), name='decoded_mean')(h)
        #rc = RecurrentContainer(readout=True, output_length=max_length) #, unroll=True, input_length=max_length)
        #rc.add(GRUCell(charset_length, name='gru_4', input_dim=charset_length))#input_shape=((latent_rep_size,)))) #((None,max_length,latent_rep_size)))) # input_shape=((latent_rep_size,max_length))))#input_shape=K.int_shape(h)))
        #rc.add(GRUCell(50, name='gru_2'))
        #rc.add(GRUCell(50, name='gru_3'))
        #rc.add(GRUCell(charset_length, name='gru_3'))
        #rc.add(Dense(charset_length, activation='softmax', name='decoded_mean'))
        #rc.add(GRUCell(charset_length, activation='softmax', name='decoded_mean'))
        #h = rc(h)
        #return TimeDistributed(Dense(charset_length, activation='softmax'), name='decoded_mean')(h)
        #rc.add(TimeDistributed(Dense(charset_length, activation='softmax'), name='decoded_mean'))
        #return TerminalGRU(charset_length, return_sequences=True, activation='softmax', name='decoded_mean')(h) # NEW!

    def save(self, filename):
        self.autoencoder.save_weights(filename)
    
    def load(self, charset, weights_file, latent_rep_size = 2, max_length=MAX_LEN, hypers = {'hidden': 200, 'dense': 200, 'conv1': 2, 'conv2': 3, 'conv3': 4}):
        self.create(charset, max_length = max_length, weights_file = weights_file, latent_rep_size = latent_rep_size, hypers = hypers)
