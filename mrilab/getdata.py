#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 15:08:38 2016

@author: modlab
"""
from os.path import exists
import cPickle as pickle
from brain import MultiBrain
from preprocessing import fnames_to_targets
import numpy as np
#import sys
#sys.path.extend(['/Users/modlab/y/projects_python/ml16_p1', '/home/modlab1/data/alex/', '/Users/modlab/Desktop/Alex'])

def pickling(traindataset,trainlabels, testdataset, testlabels, testnames, pickle_file):  
        print('traindata shape: {}  trainlabels length: {}'.format(traindataset.shape,len(trainlabels)))
        print('testdata shape {}  testlabels length: {}'.format(testdataset.shape, len(testlabels)))
        print(pickle_file)
        try:
          f = open(pickle_file, 'wb')
          save = {
            'testdataset': testdataset,
            'testlabels': testlabels,
            'traindataset': traindataset,
            'trainlabels': trainlabels,
            'testnames': testnames
            }
          pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
          f.close()
          print("finished pickling")
        except:
            print("Unable to save data to {}".format(pickle_file))
    
            
def openpickled(pickle_file):   
    with open(pickle_file, 'rb') as f:
      save = pickle.load(f)
      testdataset = save['testdataset']
      testlabels = save['testlabels']
      traindataset = save['traindataset']
      trainlabels = save['trainlabels']
      testnames = save['testnames']
      del save  # hint to help gc free up memory
      print("finished loading datasets and labels from pickled")
      
      return traindataset, trainlabels, testdataset, testlabels, testnames

      
def get_data(pickle_file,procs=1,force=False): 
    print "Load files..."
    if exists(pickle_file) and not force:
        traindataset, trainlabels, testdataset, testlabels, testnames = \
        openpickled(pickle_file)
    else:
#        dir_train = '/Users/modlab/Desktop/ml16_p1/data/set_train'
#        dir_test = '/Users/modlab/Desktop/ml16_p1/data/set_test'
        dir_train = '/home/erik/data/erik/ml16_p1/data/set_train'
        dir_test = '/home/erik/data/erik/ml16_p1/data/set_test'
#        dir_train = '/Users/Erik/Desktop/ml16_p1/data/set_train'
#        dir_test = '/Users/Erik/Desktop/ml16_p1/data/set_test'
#        # dir_train = '/home/modlab1/data/alex/ml16/set_train/'
#        # dir_test = '/home/modlab1/data/alex/ml16/set_test/'
#        datadict_train = fnames_to_targets(dir_train)
#        datadict_test = fnames_to_targets(dir_test)
#        train = MultiBrain(datadict_train, dir_train)
#        test = MultiBrain(datadict_test, dir_test)
#        train.combine_brains()
#        test.combine_brains()
#        
#        print "Smoothing images..."
#        train.smooth(fwhm=2, numproc=procs)
#        test.smooth(fwhm=2, numproc=procs)
#        
#        print "Downsizing..."
#        train.vox_brain((4, 4, 4))
#        test.vox_brain((4, 4, 4))
#        
#        print "Masking..."
#        train.cube_mask(apply=True, plot=True, x1=12, x2=30, y1=12, y2=40, z1=12, z2=30)
#        test.cube_mask(apply=True, plot=True, x1=12, x2=30, y1=12, y2=40, z1=12, z2=30)
#        # x1=50, x2=120, y1=50, y2=150, z1=50, z2=110
#        #train.ellipsoid_mask(a=40, b=60, c=40, plot=False, apply=True)
#        #test.ellipsoid_mask(a=40, b=60, c=40, plot=False, apply=True)
        
        datadict_train = fnames_to_targets(dir_train)
        datadict_test = fnames_to_targets(dir_test)
        train = MultiBrain(datadict_train, dir_train)
        test = MultiBrain(datadict_test, dir_test)
        train.combine_brains()
        test.combine_brains()
        
        print "Smoothing images..."
        train.smooth(fwhm=2)
        test.smooth(fwhm=2)
        
        print "Rescaling..."
        train.rescale(numproc=procs)
        test.rescale(numproc=procs)
        
        print "Downsampling..."
        train.vox_brain(scaletuple=(2, 2, 2), plot=False)
        test.vox_brain(scaletuple=(2, 2, 2), plot=False)
        
        print "Calculating correlation brain..."
        train.cor_brain(numproc=procs, mask=False, plot=False)
        
        #print "Calculating covariance brain..."
        #train.cov_brain(numproc=procs, plot=False)
        
        print "Compute masks..."
        train.background_mask(plot=False)
        test.background_mask(plot=False)
        
        #train.mask = (abs(train.cor.dataobj) > (np.mean(train.cor.dataobj) + 2 * np.std(train.cor.dataobj)))
        train.mask = (abs(train.cor.dataobj) > (np.mean(train.cor.dataobj) + 3 * np.std(train.cor.dataobj))).astype('int')
        test.mask = train.mask
        
        print "Apply mask..."
        train.apply_mask()
        test.apply_mask()

        pickling(train.data,train.targets, test.data, test.targets, test.names, pickle_file)
        traindataset = train.data
        trainlabels = train.targets
        testdataset = test.data
        testlabels = test.targets
        testnames = test.names
        
    return traindataset, trainlabels, testdataset, testlabels, testnames