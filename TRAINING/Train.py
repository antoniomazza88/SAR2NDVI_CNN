# -*- coding: utf-8 -*-
"""
Created in 2017

@author: mass.gargiulo & anto.mazza
"""

#############
#   A CNN-Based Fusion Method for Feature Extraction from Sentinel Data

#(http://www.mdpi.com/2072-4292/10/2/236) 
#############

from __future__ import print_function

import time
from funct_Train import *
import numpy as np
import theano
import theano.tensor as T
import lasagne
import scipy.io as sio



def main(data_folder, output_folder, identifier, num_epochs=1500):

    # Hyper-parameters
    t0,tt0,v0,rate= 0,0,0,0.5*10**(-2)


    ps = 33  # patch (linear) size
    k_1 = 9  # receptive field side - layer 1
    k_2 = 5  # receptive field side - layer 2
    k_3 = 5  # receptive field side - layer 3
    
    r = ((k_1 - 1) + (k_2 - 1) + (k_3 - 1)) / 2
    ########################################################

        # Load the dataset
    print("Loading data...")
    
    num = 1
    X_train, y_train, X_val, y_val = load_dataset(data_folder, ps, r, identifier,num) 


    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.tensor4('targets') 

    
    # Model building
    print("Building model and compiling functions...")
    network = build_cnn(input_var,X_train.shape[1])
    
    # Create loss for training
    prediction = lasagne.layers.get_output(network)
    
    loss = abs(prediction-target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training
    # Here, we'll use Stochastic Gradient Descent (SGD) with Nesterov momentum
    params = lasagne.layers.get_all_params(network, trainable=True)
    l_rate = T.scalar('learn_rate','float32')
    updates = lasagne.updates.momentum(loss, params, l_rate, momentum=0.9)
    
    # Create a loss expression for validation/testing. The crucial difference here is
    # that we do a deterministic forward pass through the network, disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = abs(test_prediction-target_var)
    test_loss = test_loss.mean()

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:8
    train_fn = theano.function([input_var,target_var, l_rate], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], test_loss)  
    
    # Finally, launch the tcraining loop.
    print("Starting training...")
    train_loss_curve = []
    val_loss_curve = []

    # We iterate over epochs:

    for epoch in range(num_epochs):
            # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 128, shuffle=False):
            inputs, targets = batch
            train_err += train_fn(inputs, targets,rate)
            train_batches += 1

            # And a full pass over the validation data:
        val_err = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 128, shuffle=False):
            inputs, targets = batch
            err = val_fn(inputs, targets)
            val_err += err
            val_batches += 1

            # Then we print the results for this epoch:

        
        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        
        t = train_err / train_batches
        v = val_err / val_batches
        print("  training loss:\t\t{:.10f}".format(t))
        print("  validation loss:\t\t{:.10f}".format(v))
        print("  gain of training loss:\t\t{:.10f}".format(t0-t))
        print("  gain validation loss:\t\t{:.10f}".format(v0-v))

        t0 = t
        v0 = v
        train_loss_curve.append(t)
        val_loss_curve.append(v)



        get_param_fn = theano.function([], params)        
        suffix = '_ID'+identifier+'_date_'+str(num)
        sio.savemat(output_folder+'loss'+suffix+'.mat',
                    {'train_loss': np.asarray(train_loss_curve), 'val_loss': np.asarray(val_loss_curve)})
                    
        np.savez(output_folder+'model'+suffix+'.npz', *get_param_fn())
        
 
if __name__ == '__main__':
    kwargs = {}
    kwargs['data_folder'] = '/DATASET/'
    kwargs['output_folder'] = '/MODEL/'
    kwargs['identifier'] = 'OPTI'
    kwargs['n_epochs'] = 10
    main(kwargs['data_folder'], kwargs['output_folder'], kwargs['identifier'], kwargs['n_epochs'])
