"""
Script with actual ML model worflow
"""
#import sys
#sys.path.extend(['/Users/modlab/y/projects_python/ml16_p1', '/home/modlab1/data/alex/', '/Users/modlab/Desktop/Alex'])

from time import time
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from getdata import get_data
#from fc_net_reg import DNN
from fc_net_class import DNN
import numpy as np


globtime = time()
procs = 12

print "Load files..."
#dir_train = '/Users/modlab/Desktop/Alex/data/set_train'
#dir_test = '/Users/modlab/Desktop/Alex/data/set_test'
dir_train = '/Volumes/Platte1/x/documents/lectures/ml16/data/set_train'
dir_test = '/Volumes/Platte1/x/documents/lectures/ml16/data/set_test'
# dir_train = '/home/modlab1/data/alex/ml16/set_train/'
# dir_test = '/home/modlab1/data/alex/ml16/set_test/'

datadict_train = fnames_to_targets(dir_train)
datadict_test = fnames_to_targets(dir_test)
train = MultiBrain(datadict_train, dir_train, sparse=True)
test = MultiBrain(datadict_test, dir_test)
train.combine_brains()
test.combine_brains()

#print "Smoothing images..."
#train.smooth(fwhm=2., numproc=procs)
#test.smooth(fwhm=2., numproc=procs)

print "Downsizing..."
train.vox_brain((3, 3, 3), plot=False)
test.vox_brain((3, 3, 3), plot=False)

print "Masking..."
#train.cube_mask(apply=True, plot=True, x1=15, x2=45, y1=17, y2=52, z1=17, z2=40)
#test.cube_mask(apply=True, plot=True, x1=17, x2=40, y1=17, y2=50, z1=17, z2=37)
# x1=50, x2=120, y1=50, y2=150, z1=50, z2=110
train.ellipsoid_mask(a=20, b=23, c=20, plot=False, apply=True)
test.ellipsoid_mask(a=20, b=23, c=20, plot=False, apply=True)

print "Training pipeline..."
#pipeline = Pipeline([('scl', StandardScaler()), ('vart', VarianceThreshold()),
#                     ('pca', PCA(n_components=100, svd_solver='randomized')),
#                     ('rgrssr', SVR(C=2., kernel='linear', verbose=1))])

pipeline = Pipeline([('scl', StandardScaler()), ('vart', VarianceThreshold()),
                     ('pca', PCA(n_components=250, svd_solver='randomized')),
                     ('rgrssr', SVR(C=2., kernel='linear'))])

pipeline.fit(traindataset, trainlabels)

print "Predicting..."
preds_test = pipeline.predict(testdataset).astype('int')

print "Saving predictions..."
ids = [int(s.split('_')[1].split('.')[0]) for s in testnames]
#print "Saving predictions..."
#ids = [int(s.split('_')[1].split('.')[0]) for s in testnames]

print(preds_test)       
df = pd.DataFrame({'ID': ids, 'Prediction': preds_test})
df.to_csv('/Users/modlab/Desktop/predictions.csv', index=False)

print("whole duration: %s" % (time() - globtime))
print "DONE!"
        

#
#print "Training pipeline..."
##pipeline = Pipeline([('scl', StandardScaler()), ('vart', VarianceThreshold()),
##                     ('pca', PCA(n_components=100, svd_solver='randomized'))])
#
#pipeline = Pipeline([('scl', StandardScaler()), ('vart', VarianceThreshold())])
#
#traindataset = pipeline.fit_transform(traindataset)
#testdataset = pipeline.fit_transform(testdataset)
#trainlabels  = np.asarray(trainlabels) 
##print trainlabels.shape
## one-hot encode labels and get number of classes
#uniques = np.unique(trainlabels)
##print uniques
#num_classes = len(uniques)
#train_labels = (uniques[np.arange(num_classes)] == trainlabels[:,None]).astype(np.float32)
##print train_labels
#
## choose network architecture
#init_archit = [traindataset.shape[1],650,150, num_classes] 
#init_activfuncs = ["input", "sigmoid","sigmoid", "logits"]
#num_steps = 1000
#MyDNN = DNN(init_archit, init_activfuncs, num_steps, num_classes)
#optimizer, train_prediction, test_prediction, graph, loss = \
#MyDNN.make_graph(traindataset, train_labels, testdataset)
## calculate
#testprediction = MyDNN.session(graph,optimizer, loss, train_prediction, test_prediction, train_labels)
## decode one-hot prediction back to single array
#testprediction = np.argmax(testprediction, 1)
#preds_test = np.asarray([uniques[i] for i in testprediction])
##print preds_test
## decode single
#print "Saving predictions..."
#ids = [int(s.split('_')[1].split('.')[0]) for s in testnames]
#       
#df = pd.DataFrame({'ID': ids, 'Prediction': preds_test})
##df.to_csv('/Users/modlab/Desktop/predictions.csv', index=False)
##df.to_csv('/Users/modlab/Desktop/ml16_p1/predictions.csv', index=False)
#df.to_csv('/home/erik/data/erik/ml16_p1/predictions.csv', index=False)
## df.to_csv('/home/modlab1/data/alex/predictions.csv', index=False)
#
#print("whole duration: %s" % (time() - globtime))
#print "DONE!"
