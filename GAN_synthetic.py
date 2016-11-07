"""Implements the long-short term memory character model.
This version vectorizes over multiple examples, but each string
has a fixed length."""

from __future__ import absolute_import
from __future__ import print_function
from builtins import range
from os.path import dirname, join
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.scipy.misc import logsumexp

from autograd.optimizers import adam, adam_state, sgd
from rnn import string_to_one_hot, one_hot_to_string,\
                build_dataset_custom, sigmoid, concat_and_multiply
import pdb
import cPickle as pickle
from copy import deepcopy
import sys, getopt

# random struct for generator
RS=npr.RandomState(0)

# initial lstm parameters for MLE generator (initialization for generator)
def init_lstm_params(input_size, state_size, output_size,
                     param_scale=0.01, rs=npr.RandomState(0)):
    def rp(*shape):
        return rs.randn(*shape) * param_scale

    return {'init cells':   rp(1, state_size),
            'init hiddens': rp(1, state_size),
            'change':       rp(input_size + state_size + 1, state_size),
            'forget':       rp(input_size + state_size + 1, state_size),
            'ingate':       rp(input_size + state_size + 1, state_size),
            'outgate':      rp(input_size + state_size + 1, state_size),
            'predict':      rp(state_size + 1, output_size)}

# initial discriminator lstm parameters
def init_discrim_params(input_size, state_size, output_size,
                     param_scale=0.01, rs=npr.RandomState(0)):
    def rp(*shape):
        return rs.randn(*shape) * param_scale

    return {'init cells':   rp(1, state_size),
            'init hiddens': rp(1, state_size),
            'change':       rp(input_size + state_size + 1, state_size),
            'forget':       rp(input_size + state_size + 1, state_size),
            'ingate':       rp(input_size + state_size + 1, state_size),
            'outgate':      rp(input_size + state_size + 1, state_size),
            'predict':      rp(state_size + 1, output_size)}

# lstm prediction for generator, this will be a sequence of softmax vectors for each input
def lstm_generate(params, inputs, TEMP, rand_hidden, rand_cells, noise_type):

    if (noise_type == 3):
        TEMP = 1

    def update_lstm(input, hiddens, cells):
        change  = np.tanh(concat_and_multiply(params['change'], input, hiddens))
        forget  = sigmoid(concat_and_multiply(params['forget'], input, hiddens))
        ingate  = sigmoid(concat_and_multiply(params['ingate'], input, hiddens))
        outgate = sigmoid(concat_and_multiply(params['outgate'], input, hiddens))
        cells   = cells * forget + ingate * change
        hiddens = outgate * np.tanh(cells)
        return hiddens, cells

    def hiddens_to_output_probs(hiddens):
        output = concat_and_multiply(params['predict'], hiddens)
        noise = npr.gumbel(loc=0.0,scale=1.0,size=output.shape)
        output = TEMP*(output + noise)
        return output - logsumexp(output, axis=1, keepdims=True) # Normalize log-probs.

    hiddens = rand_hidden
    cells   = rand_cells

    current = hiddens_to_output_probs(hiddens)
    output = [current]
    count = 1
    for input in inputs:  # Iterate over time steps.
        hiddens, cells = update_lstm(current, hiddens, cells)
        current = hiddens_to_output_probs(hiddens)
        output.append(current)
    
    return np.array(output)

# discriminator prediction (many-to-one lstm, outputs are sigmoid)
def discrim_predict(params, inputs):
    def update_lstm(input, hiddens, cells):
        change  = np.tanh(concat_and_multiply(params['change'], input, hiddens))
        forget  = sigmoid(concat_and_multiply(params['forget'], input, hiddens))
        ingate  = sigmoid(concat_and_multiply(params['ingate'], input, hiddens))
        outgate = sigmoid(concat_and_multiply(params['outgate'], input, hiddens))
        cells   = cells * forget + ingate * change
        hiddens = outgate * np.tanh(cells)
        return hiddens, cells

    def hiddens_to_output_probs(hiddens):
        output = concat_and_multiply(params['predict'], hiddens)
        return 1.0/(1.0 + np.exp(-output))

    num_sequences = inputs.shape[1]
    hiddens = np.repeat(params['init hiddens'], num_sequences, axis=0)
    cells   = np.repeat(params['init cells'],   num_sequences, axis=0)
    for input in inputs:  # Iterate over time steps.
        hiddens, cells = update_lstm(input, hiddens, cells)

    output = hiddens_to_output_probs(hiddens) 
    return output


# loss for generator
def generator_loglike(g_params, d_params, inputs, TEMP, rand_hidden, rand_cells, noise_type):

    # 1. Generate softmax outputs
    g_inputs = lstm_generate(g_params, inputs, TEMP, rand_hidden, rand_cells, noise_type)
    
    # 2. Predict log probabilities of generated outputs (real or not?)
    probs = discrim_predict(d_params, g_inputs[0:-1,:,:]) # [n,1]

    # 3. E_z log(D(G(z))) - log(1 - D(G(z)))
    n = probs.shape[0]
    ll = (1.0/n)*np.sum(np.log(probs) - np.log(1 - probs))
    

    return ll

# loss for discriminator
def discriminator_loglike(d_params, g_params, inputs, TEMP, rand_hidden, rand_cells, noise_type):
    # 1. Generate softmax outputs
    g_inputs = lstm_generate(g_params, inputs, TEMP, rand_hidden, rand_cells, noise_type)
    
    # 2. Get (noisy) inputs
    n_inputs = noise_inputs(inputs, TEMP, noise_type)

    # 3. Get discriminator predictions on (noisy) inputs
    probs_input = discrim_predict(d_params, n_inputs)
    n = probs_input.shape[0]

    # 4. E_x log(D(x))
    ll1 = (1.0/n)*np.sum(np.log(probs_input))

    # 5. Get discriminator predictions on generated points
    probs_gen   = discrim_predict(d_params, g_inputs[0:-1,:,:])

    # 6. E_z log(1-D(G(z)))
    n = probs_gen.shape[0]
    ll2 = (1.0/n)*np.sum(np.log(1.0 - probs_gen))
    
    ll = ll1 + ll2

    return ll

# this first finds the vector that leads to a softmax that is essentially 1-hot, then adds Gumbel
# noise and computes log-softmax
def noise_inputs(inputs, TEMP, noise_type):
    count = 0
    perturbed = np.empty_like(inputs)
    perturbed[:] = inputs
    perturbed[perturbed==1]=5 
    perturbed[perturbed==0]=1 
    if noise_type == 2:
        TEMP = 1
    # [5,1,...] - softmax = [ 0.91610478,  0.01677904,  ...] for 6 dim
    # for each sequence time step, add gumbel noise
    # as per this blog post: http://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/
    for pert in perturbed:  # Iterate over time steps.
        noise = npr.gumbel(loc=0.0,scale=1,size=pert.shape)
        output = TEMP*(pert + noise)
        perturbed[count] = output - logsumexp(output, axis=1, keepdims=True)
        count = count + 1
    return perturbed


def main(argv):
    # these are the possible characters we can generate for the synthetic example
    chars = "x+*-/ "
    num_chars = len(chars)

    # let's load all of our command line arguments
    # first we'll set some defaults
    STATE_SIZE = 10
    MINI_BATCH = 200
    NOISE_TYPE = 1
    LR = 0.001
    MAX_ITER = 20000
    MULT = 1
    # load any that need to be changed
    try:
        opts, args = getopt.getopt(argv, "hs:b:n:l:i:m:", ["help=", "state=", "batch=", "noise=", "lr=", "iter=", "mult="])
    except getopt.GetoptError:
        print(argv)
        print("error")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("possible options")
            print(["help=", "state=", "batch=", "noise=", "lr="])
        elif opt in ("-s", "--state"):
            STATE_SIZE = int(arg)
            print('S=' + str(STATE_SIZE))
        elif opt in ("-b", "--batch"):
            MINI_BATCH = int(arg)
            print('B=' + str(MINI_BATCH))
        elif opt in ("-n", "--noise"):
            NOISE_TYPE = int(arg)
            print('N=' + str(NOISE_TYPE))
        elif opt in ("-l", "--lr"):
            LR = float(arg)
            print('L=' + str(LR))
        elif opt in ("-i", "--iter"):
            MAX_ITER = int(arg)
            print('I=' + str(MAX_ITER))
        elif opt in ("-m", "--mult"):
            MULT = int(arg)
            print('M=' + str(MULT))


    # Learn to predict a CFG: 
    #     S -> S "+" S | S "*" S | S "-" S | S "/" S |  "x" 
    text_filename = join(dirname(__file__), 'data/train_eq_x+*-_depth5_m12.txt')
    
    train_inputs = build_dataset_custom(text_filename, sequence_length=12, #15,
                                 alphabet_size=num_chars, alphabet=chars, max_lines=5000)

    init_params = init_lstm_params(input_size=num_chars, output_size=num_chars,
                                   state_size=STATE_SIZE, param_scale=0.01)

    init_discrim_params = init_lstm_params(input_size=num_chars, output_size=1, #2,
                                   state_size=STATE_SIZE, param_scale=0.01)
    
    # we'll initialize our generator and discriminator parameters 
    g_params = init_params
    d_params = init_discrim_params

    # we have 4 different noise (temperature) settings:
    # 1. keep the temperature fixed to 1 always
    # 2. fix the temp to 1 for the inputs, vary it for the generator
    # 3. fix the temp to 1 for the generator, vary for the inputs
    # 4. vary for both
    if NOISE_TYPE == 1:
        temps = np.linspace(1,1,num=MAX_ITER)
    elif (NOISE_TYPE == 2) or (NOISE_TYPE == 3) or (NOISE_TYPE == 4):
        temps = np.linspace(0.2,1,num=MAX_ITER/2.0)

    # set up loss functions for discriminator and generator
    # these are the noise values we'll feed into the generator (changes every iteration)
    num_sequences = MINI_BATCH
    hiddens = np.repeat(RS.uniform(size=(1, STATE_SIZE)), num_sequences*MULT, axis=0) # NOTE: not multiplying by scale=0.01
    cells   = np.repeat(RS.uniform(size=(1, STATE_SIZE)), num_sequences*MULT, axis=0)

    # this is the first training batch
    train_batch = train_inputs[:,0:MINI_BATCH,:]

    # loss functions
    def discrim_loss(params, iter):
        return -discriminator_loglike(params, g_params, train_batch, TEMP, hiddens, cells, NOISE_TYPE)

    def gen_loss(params, iter):
        return -generator_loglike(params, d_params, train_batch, TEMP, hiddens, cells, NOISE_TYPE)

    # record losses over time
    gen_losses = np.zeros((MAX_ITER,))
    dis_losses = np.zeros((MAX_ITER,))

    # let's save the parameters every so often
    param_save_str = 'results/gan_params_synthetic2_S' + str(STATE_SIZE) + '_B' + str(MINI_BATCH) + '_N' + str(NOISE_TYPE) + '_L' + str(LR) + '_I' + str(MAX_ITER) + '_M' + str(MULT) + '_0.p'
    f = open(param_save_str,'wb')
    pickle.dump({'d_p': d_params, 'g_p': g_params}, f)
    f.close()

    # alternating minimization over generator and discriminator
    d_state = {}
    g_state = {}
    ix = np.arange(train_inputs.shape[1])
    for r in range(MAX_ITER):
        if r % (train_inputs.shape[1]/MINI_BATCH) == 0:
            np.random.shuffle(ix)
            start = 0
            end = MINI_BATCH

        # are we still annealing or are we done? (we stop after half of the iterations)
        if r >= len(temps):
            TEMP = temps[len(temps)-1]
        else:
            TEMP = temps[r]
        
        print('iter=' +  str(r))
        print('TEMP=' + str(TEMP))
        
        train_batch = train_inputs[:,ix[start:end],:]

        # update gradient of discriminator loss function using autograd
        discrim_loss_grad = grad(discrim_loss)

        # train discriminator
        print("Training Discriminator...")
        (d_params,d_state) = adam_state(discrim_loss_grad, d_params, d_state, step_size=LR, b1=0.5, num_iters=1)
        
        # ---- report the full discriminator training loss ---- #
        train_batch = train_inputs

        NN = train_inputs.shape[1]
        hiddens_batch = deepcopy(hiddens)
        cells_batch = deepcopy(cells)

        hiddens = np.repeat(RS.uniform(size=(1, STATE_SIZE)), NN, axis=0)
        cells   = np.repeat(RS.uniform(size=(1, STATE_SIZE)), NN, axis=0)
        print("done, loss now:", discrim_loss(d_params,0))

        # compute discriminator loss
        dis_losses[r] = discrim_loss(d_params,0)
        
        train_batch = train_inputs[:,ix[start:end],:]
        hiddens = hiddens_batch
        cells = cells_batch
        # ---- done ---- #


        # update gradient of generator loss function using autograd
        gen_loss_grad = grad(gen_loss)

        # train generator
        print("Training Generator...")
        (g_params,g_state) = adam_state(gen_loss_grad, g_params, g_state, step_size=LR, b1=0.5,num_iters=1)


        # ---- report the full generator training loss ---- #
        train_batch = train_inputs

        NN = train_inputs.shape[1]
        hiddens_batch = deepcopy(hiddens)
        cells_batch = deepcopy(cells)

        hiddens = np.repeat(RS.uniform(size=(1, STATE_SIZE)), NN, axis=0)
        cells   = np.repeat(RS.uniform(size=(1, STATE_SIZE)), NN, axis=0)

        print("done, loss now:", gen_loss(g_params,0))

        # compute generator loss
        gen_losses[r] = gen_loss(g_params,0)

        train_batch = train_inputs[:,ix[start:end],:]
        hiddens = hiddens_batch
        cells = cells_batch
        # ---- done ---- #


        # sample new noise for generator next iteration
        hiddens = np.repeat(RS.uniform(size=(1, STATE_SIZE)), num_sequences*MULT, axis=0)
        cells   = np.repeat(RS.uniform(size=(1, STATE_SIZE)), num_sequences*MULT, axis=0)

        # save losses
        loss_save_str = 'results/gan_losses_synthetic2_S' + str(STATE_SIZE) + '_B' + str(MINI_BATCH) + '_N' + str(NOISE_TYPE) + '_L' + str(LR) + '_I' + str(MAX_ITER) + '_M' + str(MULT) + '.p'
        f = open(loss_save_str,'wb')
        pickle.dump({'dis': dis_losses, 'gen': gen_losses}, f)
        f.close()

        # update batch pointers
        start = start + MINI_BATCH
        end   = end   + MINI_BATCH

        # save parameters
        if r % 100 == 0:
            param_save_str = 'results/gan_params_synthetic2_S' + str(STATE_SIZE) + '_B' + str(MINI_BATCH) + '_N' + str(NOISE_TYPE) + '_L' + str(LR) + '_I' + str(MAX_ITER) + '_M' + str(MULT) + '_' + str(r) + '.p'
            f = open(param_save_str,'wb')
            pickle.dump({'d_p': d_params, 'g_p': g_params, 'd_s': d_state, 'g_s': g_state}, f)
            f.close()
    
    param_save_str = 'results/gan_params_synthetic2_S' + str(STATE_SIZE) + '_B' + str(MINI_BATCH) + '_N' + str(NOISE_TYPE) + '_L' + str(LR) + '_I' + str(MAX_ITER) + '_M' + str(MULT) + '_' + str(MAX_ITER) + '.p'
    f = open(param_save_str,'wb')
    pickle.dump({'d_p': d_params, 'g_p': g_params, 'd_s': d_state, 'g_s': g_state}, f)
    f.close()

if __name__ == '__main__':
    main(sys.argv[1:])
