#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 17:27:10 2016

@author: modlab
"""

import tensorflow as tf
from six.moves import range
import numpy as np

#def accuracy(predictions, labels):
#    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

def gendatareg():
    """
    Make some fake train & test data for a regression DNN.
    """
    numsamples = 1000
    randmat = np.random.random((numsamples,5))*100
    labels = np.random.randint(100, size=numsamples)
    randmat[:,1] = labels*2 + np.random.randn() * 0.33
    randmat[:,2] = labels/2 + np.random.randn() * 0.33
    data = randmat
    traindataset = data[:(numsamples/2),:]
    testdataset = data[(numsamples/2):,:]
    trainlabels = labels[:(numsamples/2)]
    testlabels = labels[(numsamples/2):]
    print traindataset.shape
    print testdataset.shape
    print trainlabels.shape
    print testlabels.shape
    return traindataset,testdataset,trainlabels,testlabels
    
    
class DNN(object):
    """
    Base class for constructing a Deep Neural Network
    """
    def __init__(self,archit,activfuncs,num_steps,num_classes):
        """
        :param archit: {list} list containing network architecture
        :param activfuncs: {list} list containing activation functions to be applied
        :param num_steps: {int} number of training steps (epochs)
        :param num_classes: {int} number of classes
        """
        self.num_classes = num_classes
        self.archit = archit
        self.activfuncs = activfuncs
        self.num_steps = num_steps
        self.loginterval = num_steps/10
        self.regfactor = 0.002
#        self.learningrate = 0.01
        
    def make_graph(self,traindataset, trainlabels, testdataset):
        """
        :param data: {array} 2D np.array of the training data.
        :param labels: {array} 1D np.array of the labels corresponding to the training data
        :param data: {array} 2D np.array of the test data
        :param labels: {array} 1D np.array of the labels corresponding to the test data
        """
        graph = tf.Graph()
        with graph.as_default():
            tf_test_dataset  = tf.constant(testdataset,dtype=tf.float32)
            tf_train_dataset = tf.constant(traindataset,dtype=tf.float32)
            tf_train_labels = tf.constant(trainlabels,dtype=tf.float32)
            
            weights  = [] # list containing tensorflow tensors of the weights for each layer
            biases   = [] # list containing tensorflow tensors of the biases for each layer
            for i in range(1,len(self.archit)):
                weights.append(tf.Variable(tf.truncated_normal([self.archit[i-1], self.archit[i]],stddev=0.1)))
                biases.append(tf.Variable(tf.zeros([self.archit[i]])))
            modes = ["train","test"]
            for mode in modes:
                matrices = [] # list containing tensorflow tensors of the output-value matrices for each layer
                if   mode == "train":
                   matrices.append(tf_train_dataset)  # first matrix for the computations is the training data matrix 
                elif mode == "test":
                   matrices.append(tf_test_dataset)  # first matrix for the computations is the test data matrix   
                else:
                    print("invalid computation mode in make_graph")
                    
                for i in range(1,len(self.archit)):    
                    if self.activfuncs[i] == "logits": # i.e. do nothing -> activation fct. has yet to be applied to this layer.
                        matrices.append(tf.matmul(matrices[i-1], weights[i-1]) + biases[i-1])
                    elif self.activfuncs[i] == "relu":
                        matrices.append(tf.nn.relu(tf.matmul(matrices[i-1], weights[i-1]) + biases[i-1]))
                    elif self.activfuncs[i] == "tanh": 
                        matrices.append(tf.tanh(tf.matmul(matrices[i-1], weights[i-1]) + biases[i-1])) 
                    elif self.activfuncs[i] == "softplus": 
                        matrices.append(tf.nn.softplus(tf.matmul(matrices[i-1], weights[i-1]) + biases[i-1]))
                    elif self.activfuncs[i] == "sigmoid": 
                        matrices.append(tf.sigmoid(tf.matmul(matrices[i-1], weights[i-1]) + biases[i-1]))
                    else:
                        print("missing activation function")
                
                for i in range(1,len(self.archit)):
                    print(self.activfuncs[i])
                for i in range(len(weights)):
                    print("weights {}:{}".format(i, weights[i].get_shape().as_list()))
                
                if mode == "train":
                    loss = tf.reduce_sum(tf.squared_difference(matrices[len(self.archit)-1],tf_train_labels))
                    for i in range(len(self.archit)):  # add error terms for each weightmatrix in the net
                        loss += 0.5*self.regfactor*tf.nn.l2_loss(weights[i-1])
                    optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)
                    train_prediction = matrices[len(self.archit)-1]
                elif mode == "test":
                    test_prediction = matrices[len(self.archit)-1]
                else:
                    print("invalid computation mode in make_graph(...): Must be either train or test")
                
        return optimizer, train_prediction, test_prediction, graph, loss
        
    def session(self,graph,optimizer, loss, train_prediction, test_prediction, test_labels,train_labels): 
        with tf.Session(graph=graph) as session:
            tf_test_labels = tf.constant(test_labels,dtype=tf.float32)
            tf.initialize_all_variables().run()
            print("Initialized")
            for step in range(self.num_steps):
                _, l, predictions = session.run([optimizer, loss, train_prediction])
                if (step % self.loginterval == 0):
                    print('Loss at step %d: %f' % (step, l))
                    print("current training prediction:")
                    print(train_prediction.eval())
#            testloss = ((test_prediction.eval()-testlabels)**2).mean()
            tf_testloss = tf.reduce_sum(tf.squared_difference(test_prediction,tf_test_labels))
            testloss = tf_testloss.eval()
            testprediction = test_prediction.eval()
            testpred = testprediction.reshape((testprediction.shape[0])).astype(int)
            
        return testloss,testpred
        
        
if __name__ == '__main__':
    traindataset,testdataset,trainlabels,testlabels = gendatareg()
    num_classes = 1
    # choose network architecture
    init_archit = [traindataset.shape[1],100, num_classes] 
    init_activfuncs = ["input", "sigmoid", "logits"]
    num_steps = 1000
    print("train labels")
    print(trainlabels)
    print("test labels")
    print(testlabels)
            
    MyDNN = DNN(init_archit, init_activfuncs, num_steps, num_classes)
    optimizer, train_prediction, test_prediction, graph, loss = \
    MyDNN.make_graph(traindataset, trainlabels, testdataset)
    # calculate
#    testloss = MyDNN.session(graph,optimizer, loss, train_prediction, test_prediction, test_labels, train_labels)
    testloss,testprediction = MyDNN.session(graph,optimizer, loss, train_prediction, test_prediction, testlabels, trainlabels)
    print("testloss: {}\n testprediction:{}".format(testloss,testprediction))
