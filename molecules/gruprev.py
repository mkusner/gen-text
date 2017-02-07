from keras.layers.recurrent import GRU
from keras.layers.recurrent import time_distributed_dense
from keras import backend as K
import numpy as np
import pdb
import tensorflow as tf

import the_grammar as G

masks_K      = K.variable(G.masks)
ind_of_ind_K = K.variable(G.ind_of_ind)
rhs_map_K    = K.variable(G.rhs_map_sparse)


# TODO: DOESNT WORK BECAUSE NEED TO KEEP TRACK OF PRIOR NON-TERMINALS!
# Right now I can't get training data in........ teacher forcing is tricky but can try using sampling for training????? Well actually can't differentiate....
# will only be run during training
def mask_samples(logits, x_true, DIM):
    # if training, x_true will be training data
    most_likely = K.argmax(x_true) # pick productions
    ix2 = tf.expand_dims(tf.gather(ind_of_ind_K, most_likely),1) # index ind_of_ind with res
    ix2 = tf.cast(ix2, tf.int32) # cast indices as ints 
    M2 = tf.gather_nd(masks_K, ix2) # get slices of masks_K with indices
    M3 = tf.reshape(M2, [-1,DIM])#K.int_shape(x_pred)) # reshape them
    P2 = tf.mul(K.exp(logits),M3) # apply them to the exp-predictions
    P2 = tf.div(P2,K.sum(P2,axis=-1,keepdims=True)) # normalize predictions
    return P2


    #most_likely = tf.reshape(most_likely,[-1]) # flatten most_likely
    #ix2 = tf.expand_dims(tf.gather(ind_of_ind_K, most_likely),1) # index ind_of_ind with res
    #ix2 = tf.cast(ix2, tf.int32) # cast indices as ints 
    #M2 = tf.gather_nd(masks_K, ix2) # get slices of masks_K with indices
    #M3 = tf.reshape(M2, [-1,MAX_LEN,DIM])#K.int_shape(x_pred)) # reshape them
    #P2 = tf.mul(K.exp(logits),M3) # apply them to the exp-predictions
    #P2 = tf.div(P2,K.sum(P2,axis=-1,keepdims=True)) # normalize predictions
    #return P2



def cond_sample_np(x, STACK, POINT, masks, rhs_map_sparse): # (n,d)
    #pdb.set_trace()
    #x = K.eval(x)
    shape = x.shape # (n,d)
    samples = np.zeros((shape[0],shape[1]))
    for i in range(shape[0]):
        if POINT[i] == -1: # this check needs to be done in keras, this means we are done
            samples[i,-1] = 1
            continue
        # 1. pop current nt off stack
        current_nt = STACK[i,POINT[i]]
        POINT[i] = POINT[i]-1 
        the_mask = masks[current_nt]
        #where_zero = np.where(the_mask == 0)[0]
        #where = tf.equal(the_mask, zero)
        the_mask[the_mask == 0] = -999 # hack to deal with gumbel noise making things negative
        masked = x[i] * the_mask
    
        #softmax = masked / K.sum(masked, axis=-1)
        # find tensorflow code for discrete distribution sampling or do gumbel trick
        G = np.random.gumbel(size=x[i].shape)
        noise_masked = G + masked
        choice = np.argmax(noise_masked)
        
        # instead of making 1-hot, just put 1 in right place
        # (if stack is empty then we will place nothing
        # then we will look for all zero columns and put 1's at end, either in Keras or as post-processing        
        samples[i,choice] = 1
        
        new_nts = np.where(rhs_map_sparse[choice] == 1)[0]
        #new_nts = rhs_map_np[choice]
        len_nts = len(new_nts)
        #pdb.set_trace()
        if len_nts == 0:
            continue

        STACK[i,POINT[i]+1:POINT[i]+1+len_nts] = np.flipud(new_nts)
        POINT[i] = POINT[i]+len_nts

    return [samples, STACK, POINT]




class GRUPrev(GRU):
    def __init__(self, output_dim, dim, X,  **kwargs):
        super(GRUPrev, self).__init__(output_dim, **kwargs)

        self.dim = dim
        self.X = X
        shape = K.int_shape(self.X)

        self.STACK = K.variable(np.zeros((shape[0],shape[1])))
        self.POINT = K.variable(np.zeros((shape[0])))


    def build(self, input_shape):
        self.Y = self.add_weight((self.output_dim, self.output_dim),
                initializer=self.init,
                name='{}_Y'.format(self.name))
        super(GRUPrev, self).build(input_shape)
        if self.stateful:
            self.reset_states()
        else:
            # initial states: 2 all-zero tensors of shape (output_dim)
            self.states = [None, None]



    def preprocess_input(self, x):
        #self.STACK[:] = 0.0 # when we see new data, zero out stack and pointer
        #self.POINT[:] = 0.0

        shape = K.int_shape(self.X)
        self.STACK = K.variable(np.zeros((shape[0],shape[1])))
        self.POINT = K.variable(np.zeros((shape[0])))

        if self.consume_less == 'cpu':
            input_shape = K.int_shape(x)
            input_dim = input_shape[2]
            timesteps = input_shape[1]
            x_z = time_distributed_dense(x, self.W_z, self.b_z, self.dropout_W,
                                        input_dim, self.output_dim, timesteps)
            x_r = time_distributed_dense(x, self.W_r, self.b_r, self.dropout_W,
                                        input_dim, self.output_dim, timesteps)
            x_h = time_distributed_dense(x, self.W_h, self.b_h, self.dropout_W,
                                        input_dim, self.output_dim, timesteps)
            to_return = K.concatenate([x_z, x_r, x_h], axis=2)
        else:
            to_return = x
        return K.concatenate([self.X, to_return], axis=-1)

    def step(self, x, states):
        train_data = x[:,0:self.dim]
        x = x[:,self.dim:] 

        h_tm1 = states[0]  # previous memory (from previous GRU)
        from_last_time = states[1] # previous output (from last timestep)
        B_U = states[2]  # dropout matrices for recurrent units
        B_W = states[3]

        if self.consume_less == 'gpu':

            matrix_x = K.dot(x * B_W[0], self.W) + self.b
            matrix_inner = K.dot(h_tm1 * B_U[0], self.U[:, :2 * self.output_dim])

            x_z = matrix_x[:, :self.output_dim]
            x_r = matrix_x[:, self.output_dim: 2 * self.output_dim]
            inner_z = matrix_inner[:, :self.output_dim]
            inner_r = matrix_inner[:, self.output_dim: 2 * self.output_dim]

            z = self.inner_activation(x_z + inner_z)
            r = self.inner_activation(x_r + inner_r)

            x_h = matrix_x[:, 2 * self.output_dim:]
            inner_h = K.dot(r * h_tm1 * B_U[0], self.U[:, 2 * self.output_dim:])
            # NEW! don't do dropout for now
            prev_h = K.dot(from_last_time, self.Y)

            hh = self.activation(x_h + inner_h + prev_h)
        else:
            if self.consume_less == 'cpu':
                x_z = x[:, :self.output_dim]
                x_r = x[:, self.output_dim: 2 * self.output_dim]
                x_h = x[:, 2 * self.output_dim:]
            elif self.consume_less == 'mem':
                x_z = K.dot(x * B_W[0], self.W_z) + self.b_z
                x_r = K.dot(x * B_W[1], self.W_r) + self.b_r
                x_h = K.dot(x * B_W[2], self.W_h) + self.b_h
            else:
                raise ValueError('Unknown `consume_less` mode.')
            z = self.inner_activation(x_z + K.dot(h_tm1 * B_U[0], self.U_z))
            r = self.inner_activation(x_r + K.dot(h_tm1 * B_U[1], self.U_r))

            # NEW! don't do dropout for now
            prev_h = K.dot(from_last_time, self.Y)

            hh = self.activation(x_h + K.dot(r * h_tm1 * B_U[2], self.U_h) + prev_h)
        h = z * h_tm1 + (1 - z) * hh
        # make sure to not always use training data! HERERERE TODO
        #to_next_time = mask_samples(h, train_data, self.dim)
        if K.learning_phase() == 0:
            [A,B,C] = tf.py_func(cond_sample_np, [h, self.STACK, self.POINT, masks_K, rhs_map_K], tf.float32)
            SAMP = list_of_t[0]
            self.STACK = list_of_t[1]
            self.POINT = list_of_t[2]


        to_next_time = K.in_train_phase(mask_samples(h, train_data, self.dim),SAMP) 
        #return h, [h, to_next_time]
        return to_next_time, [h, to_next_time]
