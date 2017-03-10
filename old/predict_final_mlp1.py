# -*- coding: utf-8 -*-
"""
Final training and prediction workflow for MLP1 Group **mrilab**

.. author:: Alex Müller, Erik Gawehn
"""
import os
import sys

sys.path.insert(0, os.path.dirname('../'))  # path extension for mrilab module import

from time import time

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from mrilab.brain import MultiBrain
from mrilab.preprocessing import fnames_to_targets


globtime = time()

# put location of training and test data images (directory path) as well as output filename

outfile = '/Volumes/Platte1/x/documents/lectures/ml16/MLP1_mrilab/final_submission.csv'
dir_train = '/Volumes/Platte1/x/documents/lectures/ml16/data_1/set_train'
dir_test = '/Volumes/Platte1/x/documents/lectures/ml16/data_1/set_test'

# read file names and combine brain images into one 4D brain
datadict_train = fnames_to_targets(dir_train)  # combining filenames with age values
datadict_test = fnames_to_targets(dir_test)
train = MultiBrain(datadict_train, dir_train)  # MultiBrain class with all needed methods
test = MultiBrain(datadict_test, dir_test)
train.combine_brains()  # appending all images in one Nifti object
test.combine_brains()

print "Downsizing images by averaging voxels in small boxes..."
train.vox_brain(scaletuple=(2, 2, 2), plot=False)
test.vox_brain(scaletuple=(2, 2, 2), plot=False)

print "Calculating correlation of voxels with age..."
train.cor_brain(numproc=12, plot=False, mask=False)
# pick all voxels with a higher absolute correlation than mean + 2* std
train.mask = (abs(train.cor.dataobj) > (np.mean(train.cor.dataobj) + 2 * np.std(train.cor.dataobj))).astype('int')
test.mask = train.mask

print "Reduce image to correlated voxels..."
train.apply_mask()
test.apply_mask()

print "Final data dimension:"
print test.data.shape[1]

print "Training pipeline..."
# 1) Standard Scaling to zero mean and unit variance
# 2) Removing low variance voxels
# 3) Linear Support Vector Regression (SVR) with C=2
pipeline = Pipeline([('scl', StandardScaler()),
                     ('rgrssr', SVR(C=2., kernel='linear'))])

pipeline.fit(train.data, train.targets)

print "Predicting test data..."
preds_test = pipeline.predict(test.data).astype('int')

print "Saving predictions..."
ids = [int(s.split('_')[1].split('.')[0]) for s in test.names]  # get the IDs from the file names

print "Age predictions:"
df = pd.DataFrame({'ID': ids, 'Prediction': preds_test})
print df
print "\nSaving predictions to %s" % outfile
df.to_csv(outfile, index=False)

print("whole duration: %s" % (time() - globtime))
print "\nDONE!"
