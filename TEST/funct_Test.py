from __future__ import print_function

import os

import numpy as np
import lasagne
import gdal


def load_input(path,identity,num):
    dir_list = os.listdir(path)
    dir_list.sort()
    for file in dir_list:
        if file[4:6] == 'VH'and file[2] == str(num) :
            
            vh0_file =file 
            vv0_file = '00' + str(num) + '_VV'+ vh0_file[6:]
            ndvi0_file = '00' + str(num) + '_NDVI'+ vh0_file[6:]


            vh1_file = '00' + str(num+1) + '_VH'+ vh0_file[6:]
            vv1_file = '00' + str(num+1) + '_VV' + vh0_file[6:]
            
            vh2_file = '00' + str(num+2) + '_VH'+ vh0_file[6:]
            vv2_file = '00' + str(num+2) + '_VV' + vh0_file[6:]
            ndvi2_file = '00' + str(num+2) + '_NDVI'+ vh0_file[6:]
            dem_file = 'DEM.tif'
            dataset = gdal.Open(path + vh0_file, gdal.GA_ReadOnly)
            vh0 = dataset.ReadAsArray()
            dataset = None
            dataset = gdal.Open(path+vv0_file, gdal.GA_ReadOnly)
            vv0 = dataset.ReadAsArray()
            dataset = None
            dataset = gdal.Open(path+ndvi0_file, gdal.GA_ReadOnly)
            ndvi0 = dataset.ReadAsArray()
            dataset = None
                 
            dataset = gdal.Open(path + vh1_file, gdal.GA_ReadOnly)
            vh1 = dataset.ReadAsArray()
            dataset = None
            dataset = gdal.Open(path+vv1_file, gdal.GA_ReadOnly)
            vv1 = dataset.ReadAsArray()
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
            dataset = gdal.Open(path+dem_file, gdal.GA_ReadOnly)
            dem = dataset.ReadAsArray()
            dataset = None
            s1,s2 = ndvi0.shape
            
            if identity == 'SOPTII':
                B1 = 8
                x = np.ndarray(shape=(1,B1, s1, s2), dtype='float32')
                x[0,0,:,:] = vh0
                x[0,1,:,:] = vv0
                x[0,2,:,:] = vh1
                x[0,3,:,:] = vv1
                x[0,4,:,:] = vh2
                x[0,5,:,:] = vv2
                x[0,6,:,:] = ndvi0
                x[0,7,:,:] = ndvi2
                
            elif identity == 'SOPTIIp':
                B1 = 9
                x = np.ndarray(shape=(1,B1, s1, s2), dtype='float32')
                x[0,0,:,:] = vh0
                x[0,1,:,:] = vv0
                x[0,2,:,:] = vh1
                x[0,3,:,:] = vv1
                x[0,4,:,:] = vh2
                x[0,5,:,:] = vv2
                x[0,6,:,:] = ndvi0
                x[0,7,:,:] = ndvi2
                x[0,8,:,:] = dem
                
            elif identity == 'SOPTI':
                B1 = 5
                x = np.ndarray(shape=(1,B1, s1, s2), dtype='float32')
                x[0,0,:,:] = vh0
                x[0,1,:,:] = vv0
                x[0,2,:,:] = vh1
                x[0,3,:,:] = vv1
                x[0,4,:,:] = ndvi0
            elif identity == 'SOPTIp':
                B1 = 6
                x = np.ndarray(shape=(1,B1, s1, s2), dtype='float32')
                x[0,0,:,:] = vh0
                x[0,1,:,:] = vv0
                x[0,2,:,:] = vh1
                x[0,3,:,:] = vv1
                x[0,4,:,:] = ndvi0
                x[0,5,:,:] = dem

                
            elif identity == 'SAR':
                B1 = 2
                x = np.ndarray(shape=(1,B1, s1, s2), dtype='float32')
                x[0,0,:,:] = vh0
                x[0,1,:,:] = vv0

            elif identity == 'SARp':
                B1 = 3
                x = np.ndarray(shape=(1,B1, s1, s2), dtype='float32')
                x[0,0,:,:] = vh0
                x[0,1,:,:] = vv0
                x[0,2,:,:] = dem
                
            elif identity == 'OPTI':
                B1 = 1
                x = np.ndarray(shape=(1,B1, s1, s2), dtype='float32')
                x[0,0,:,:] = ndvi0
            elif identity == 'OPTII':
                B1 = 2
                x = np.ndarray(shape=(1,B1, s1, s2), dtype='float32')
                x[0,0,:,:] = ndvi0
                x[0,1,:,:] = ndvi2
            else:
                print('Insert the correct identifier. You must choose among these: \n - SOPTIIp \n - SOPTII \n - SOPTIp \n - SOPTI \n - OPTII \n - OPTI \n - SARp \n - SAR')
                quit()


            x = x[:,:,530:1000,4300:4750]
            
            return x
            

def build_cnn(input_var=None,bands=None):
      network = lasagne.layers.InputLayer(shape=(None,bands,None,None),input_var=input_var)     
      network = lasagne.layers.Conv2DLayer(network, num_filters=48, filter_size=(9,9),nonlinearity=lasagne.nonlinearities.rectify)
      network = lasagne.layers.Conv2DLayer(network, num_filters=32, filter_size=(5,5),nonlinearity=lasagne.nonlinearities.rectify)
      network = lasagne.layers.Conv2DLayer(network, num_filters=1, filter_size=(5,5))
      return network
 

