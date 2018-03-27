from __future__ import print_function

import os

import numpy as np
import lasagne
import gdal
import random
import sys


def build_cnn(input_var=None,bands=None):
     network = lasagne.layers.InputLayer(shape=(None,bands,None,None),input_var=input_var) #Patch sizes varying between train-val and test     
     network = lasagne.layers.Conv2DLayer(network, num_filters=48, filter_size=(9,9), W=lasagne.init.Normal(std = 0.001,mean = 0),nonlinearity=lasagne.nonlinearities.rectify)
     network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(5,5),W=lasagne.init.Normal(std = 0.001,mean = 0),nonlinearity=lasagne.nonlinearities.rectify)
     network = lasagne.layers.Conv2DLayer(network, num_filters=1, filter_size=(5,5),W=lasagne.init.Normal(std = 0.001,mean = 0))
     return network

def iterate_minibatches(inputs, targets, batchsize, shuffle=False): 
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]



def load_dataset(dataset_folder, patch_side, border_width,identity, num): 

    #############
    # short names
    path, ps, r = dataset_folder, patch_side, border_width

    dir_list = os.listdir(path)
    dir_list.sort()
    B = 9
    x_train = np.ndarray(shape=(0, B, ps, ps), dtype='float32')
    y_train = np.ndarray(shape=(0, 1, ps-2*r, ps-2*r), dtype='float32')
    x_val = np.ndarray(shape=(0, B, ps, ps), dtype='float32')
    y_val = np.ndarray(shape=(0, 1, ps-2*r, ps-2*r), dtype='float32')

    for file in dir_list:
        if file[4:6] == 'VH' and int(file[2:3])==num:
            vh0_file =file 
            vv0_file = '00' + str(num) + '_VV'+ vh0_file[6:]
            ndvi0_file = '00' + str(num) + '_NDVI'+ vh0_file[6:]
            mask0_file = '00' + str(num) + '_MASK'+ vh0_file[6:]
            vh_file = '00' + str(num+1) + '_VH'+ vh0_file[6:]
            vv_file = '00' + str(num+1) + '_VV' + vh0_file[6:]
            ndvi_file = '00' + str(num+1) + '_NDVI'+ vh0_file[6:]
            mask_file = '00' + str(num+1) + '_MASK'+ vh0_file[6:]            
            vh2_file = '00' + str(num+2) + '_VH'+ vh0_file[6:]
            vv2_file = '00' + str(num+2) + '_VV' + vh0_file[6:]
            ndvi2_file = '00' + str(num+2) + '_NDVI'+ vh0_file[6:]
            mask2_file = '00' + str(num+2) + '_MASK'+ vh0_file[6:]            
            dem_file = 'DEM' + vh0_file[6:]
            dataset = gdal.Open(path + vh0_file, gdal.GA_ReadOnly)
            vh0 = dataset.ReadAsArray()
            dataset = None
            dataset = gdal.Open(path+vv0_file, gdal.GA_ReadOnly)
            vv0 = dataset.ReadAsArray()
            dataset = None
            dataset = gdal.Open(path+ndvi0_file, gdal.GA_ReadOnly)
            ndvi0 = dataset.ReadAsArray()
            dataset = None
            dataset = gdal.Open(path+mask0_file, gdal.GA_ReadOnly)
            mask0 = dataset.ReadAsArray()
            dataset = None            
            dataset = gdal.Open(path + vh_file, gdal.GA_ReadOnly)
            vh = dataset.ReadAsArray()
            dataset = None
            dataset = gdal.Open(path+vv_file, gdal.GA_ReadOnly)
            vv = dataset.ReadAsArray()
            dataset = None
            dataset = gdal.Open(path+ndvi_file, gdal.GA_ReadOnly)
            ndvi = dataset.ReadAsArray()
            dataset = None
            dataset = gdal.Open(path+mask_file, gdal.GA_ReadOnly)
            mask = dataset.ReadAsArray()
            dataset = None                        
            dataset = gdal.Open(path + vh2_file, gdal.GA_ReadOnly)
            vh2 = dataset.ReadAsArray()
            dataset = None
            dataset = gdal.Open(path+vv2_file, gdal.GA_ReadOnly)
            vv2 = dataset.ReadAsArray()
            dataset = None
            dataset = gdal.Open(path+ndvi2_file, gdal.GA_ReadOnly)
            ndvi2 = dataset.ReadAsArray()
            dataset = None
            dataset = gdal.Open(path+mask2_file, gdal.GA_ReadOnly)
            mask2 = dataset.ReadAsArray()
            dataset = None       
            dataset = gdal.Open(path + dem_file, gdal.GA_ReadOnly)
            dem = dataset.ReadAsArray()
            dataset = None

            mask0[530:1000,4300:4750]=1
            mask[530:1000,4300:4750]=1
            mask2[530:1000,4300:4750]=1
   
            
            
            [s1, s2] = ndvi0.shape
            p = []
            for y in range(1,s1-ps+1,r):
                for x in range(1,s2-ps+1,r):
                    Mk = mask[y:y+ps, x:x+ps]
                    Mk0 = mask0[y:y+ps,x:x+ps]
                    Mk2 = mask2[y:y+ps,x:x+ps]
                    if  Mk0.sum() == 0 and Mk.sum()== 0 and Mk2.sum()== 0:
                        p.append([y,x])

            random.shuffle(p)
            
            P = 19000

            p_train, p_val = p[:int(0.8*P)], p[int(0.8*P):int(P)]

            x_train_k = np.ndarray(shape=(len(p_train), B, ps, ps), dtype='float32')
            y_train_k = np.ndarray(shape=(len(p_train), 1, ps-2*r, ps-2*r), dtype='float32')
            n = 0
            for patch in p_train:
                y0, x0 = patch[0], patch[1]
                x_train_k[n,0,:,:] = vh0[y0:y0+ps,x0:x0+ps]
                x_train_k[n,1,:,:] = vv0[y0:y0+ps,x0:x0+ps]
                x_train_k[n,2,:,:] = vh[y0:y0+ps,x0:x0+ps]
                x_train_k[n,3,:,:] = vv[y0:y0+ps,x0:x0+ps]
                x_train_k[n,4,:,:] = vv2[y0:y0+ps,x0:x0+ps]
                x_train_k[n,5,:,:] = vh2[y0:y0+ps,x0:x0+ps]
                x_train_k[n,6,:,:] = ndvi0[y0:y0+ps,x0:x0+ps]
                x_train_k[n,7,:,:] = ndvi2[y0:y0+ps,x0:x0+ps]
                x_train_k[n,8,:,:] = dem[y0:y0+ps,x0:x0+ps]
                
                y_train_k[n, 0, :, :] = ndvi[y0+r:y0+ps-r, x0+r:x0+ps-r]
                n = n + 1
            x_train = np.concatenate((x_train, x_train_k))
            y_train = np.concatenate((y_train, y_train_k))

            x_val_k = np.ndarray(shape=(len(p_val), B, ps, ps), dtype='float32')
            y_val_k = np.ndarray(shape=(len(p_val), 1, ps-2*r, ps-2*r), dtype='float32')
            n = 0
            for patch in p_val:
                y0, x0 = patch[0], patch[1]
                x_val_k[n,0,:,:] = vh0[y0:y0+ps,x0:x0+ps]
                x_val_k[n,1,:,:] = vv0[y0:y0+ps,x0:x0+ps]
                x_val_k[n,2,:,:] = vh[y0:y0+ps,x0:x0+ps]
                x_val_k[n,3,:,:] = vv[y0:y0+ps,x0:x0+ps]
                x_val_k[n,4,:,:] = vv2[y0:y0+ps,x0:x0+ps]
                x_val_k[n,5,:,:] = vh2[y0:y0+ps,x0:x0+ps]
                x_val_k[n,6,:,:] = ndvi0[y0:y0+ps,x0:x0+ps]
                x_val_k[n,7,:,:] = ndvi2[y0:y0+ps,x0:x0+ps]
                x_val_k[n,8,:,:] = dem[y0:y0+ps,x0:x0+ps]

                y_val_k[n, 0, :, :] = ndvi[y0+r:y0+ps-r, x0+r:x0+ps-r]
                n = n + 1
            x_val = np.concatenate((x_val, x_val_k))
            y_val = np.concatenate((y_val, y_val_k))
            
            if identity == 'SOPTII':
                B1 = 8
                x_val_temp = np.ndarray(shape=(len(p_val), B1, ps, ps), dtype='float32')
                x_train_temp = np.ndarray(shape=(len(p_train), B1, ps, ps), dtype='float32')
                x_train_temp = x_train[:,:8,:,:]
                x_val_temp = x_val[:,:8,:,:]
            elif identity == 'SOPTIIp':
                B1 = 9
                x_val_temp = np.ndarray(shape=(len(p_val), B1, ps, ps), dtype='float32')
                x_train_temp = np.ndarray(shape=(len(p_train), B1, ps, ps), dtype='float32')
                x_train_temp = x_train[:,:,:,:]
                x_val_temp = x_val[:,:,:,:]
            elif identity == 'SOPTI':
                B1 = 5
                x_val_temp = np.ndarray(shape=(len(p_val), B1, ps, ps), dtype='float32')
                x_train_temp = np.ndarray(shape=(len(p_train), B1, ps, ps), dtype='float32')
                x_train_temp[:,:4,:,:] = x_train[:,:4,:,:]
                x_val_temp[:,:4,:,:] = x_val[:,:4,:,:]

                x_train_temp[:,4,:,:] = x_train[:,6,:,:]  
                x_val_temp[:,4,:,:] = x_val[:,6,:,:]            
            elif identity == 'SOPTIp':
                B1 = 6
                x_val_temp = np.ndarray(shape=(len(p_val), B1, ps, ps), dtype='float32')
                x_train_temp = np.ndarray(shape=(len(p_train), B1, ps, ps), dtype='float32')
                x_train_temp[:,:4,:,:] = x_train[:,:4,:,:]
                x_val_temp[:,:4,:,:] = x_val[:,:4,:,:]

                x_train_temp[:,4,:,:] = x_train[:,6,:,:]  
                x_val_temp[:,4,:,:] = x_val[:,6,:,:] 
                x_train_temp[:,5,:,:] = x_train[:,8,:,:]  
                x_val_temp[:,5,:,:] = x_val[:,8,:,:]    
                
            elif identity == 'SAR':
                B1 = 2
                x_val_temp = np.ndarray(shape=(len(p_val), B1, ps, ps), dtype='float32')
                x_train_temp = np.ndarray(shape=(len(p_train), B1, ps, ps), dtype='float32')
                x_train_temp[:,:,:,:] = x_train[:,2:4,:,:]
                x_val_temp[:,:,:,:] = x_val[:,2:4,:,:]
            elif identity == 'SARp':
                B1 = 3
                x_val_temp = np.ndarray(shape=(len(p_val), B1, ps, ps), dtype='float32')
                x_train_temp = np.ndarray(shape=(len(p_train), B1, ps, ps), dtype='float32')
                x_train_temp[:,:2,:,:] = x_train[:,2:4,:,:]
                x_val_temp[:,:2,:,:] = x_val[:,2:4,:,:]
                x_train_temp[:,2,:,:] = x_train[:,8,:,:]
                x_val_temp[:,2,:,:] = x_val[:,8,:,:]
                
            elif identity == 'OPTI':
                B1 = 1
                x_val_temp = np.ndarray(shape=(len(p_val), B1, ps, ps), dtype='float32')
                x_train_temp = np.ndarray(shape=(len(p_train), B1, ps, ps), dtype='float32')
                x_train_temp[:,0,:,:] = x_train[:,6,:,:]
                x_val_temp[:,0,:,:] = x_val[:,6,:,:]
                
            elif identity == 'OPTII':
                B1 = 2
                x_val_temp = np.ndarray(shape=(len(p_val), B1, ps, ps), dtype='float32')
                x_train_temp = np.ndarray(shape=(len(p_train), B1, ps, ps), dtype='float32')
                x_train_temp[:,:,:,:] = x_train[:,6:8,:,:]
                x_val_temp[:,:,:,:] = x_val[:,6:8,:,:]
            else:
                print('Insert the correct identifier. You must choose among these: \n - SOPTIIp \n - SOPTII \n - SOPTIp \n - SOPTI \n - OPTII \n - OPTI \n - SARp \n - SAR')
                quit()

            vh0, vv0, ndvi0,vh, vv, ndvi,vh2, vv2, ndvi2,dem = None, None, None, None, None, None, None, None, None, None
    return x_train_temp, y_train, x_val_temp, y_val

