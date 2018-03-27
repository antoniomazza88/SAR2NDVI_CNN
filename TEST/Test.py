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

from funct_Test import *
import numpy as np
import theano
import theano.tensor as T
import lasagne
from PIL import Image



def main(data_folder, model_folder, output_folder, identifier):

        # Load the dataset

    print("Loading data...")
    num = 1
    x = load_input(data_folder,identifier,num)
    
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    prediction = T.tensor4('targets') 
    
    # Model building
    print("Building model and compiling functions...")
    network = build_cnn(input_var,x.shape[1])
#    identifier = identifier +'_date_'+str(num)
    with np.load(model_folder+'model_ID'+ identifier +'.npz') as g:
        param_values = [g['arr_%d' % i] for i in range(len(g.files))]
    lasagne.layers.set_all_param_values(network, param_values)

     
    prediction = lasagne.layers.get_output(network, deterministic=True)
    
    
    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    test_fn = theano.function([input_var], prediction)#, allow_input_downcast=True)#,n0]
    pred_err = test_fn(x)
    ndvi1 = pred_err[0,0,:,:]
    
    
    
    im = output_folder + identifier + '_NDVI_PRED.tif'
    ndvi1_array = np.asarray(ndvi1)
    ndvi1_array = Image.fromarray(ndvi1_array, mode='F')
    ndvi1_array.save(im, "TIFF")


if __name__ == '__main__':
    kwargs = {}
    kwargs['data_folder'] = './DATASET/'
    kwargs['model_folder'] = './MODEL/'
    kwargs['output_folder'] = './IMAGES/'
    kwargs['identifier'] = 'OPTII'
    main(kwargs['data_folder'], kwargs['model_folder'], kwargs['output_folder'], kwargs['identifier'])

