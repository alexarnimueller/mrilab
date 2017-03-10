#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 18:46:45 2016

@author: modlab
"""

import tensorflow as tf
from six.moves import range
import numpy as np
#from modlamp.sequences import Helices,Random
#from modlamp.descriptors import PeptideDescriptor
from sklearn.model_selection import train_test_split

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

#def gendata():
#    '''
#    Make some fake train & test data for the DNN.
#    '''
#    helsamples = 100
#    randsamples = 100
#    hel = Helices(30,40,helsamples)
#    rand = Random(30,40,randsamples)
#    hel.generate_helices()
#    rand.generate_sequences()
#    descr_hel = PeptideDescriptor(hel.sequences,'pepcats')
#    descr_hel.calculate_crosscorr(7)
#    descr_rand = PeptideDescriptor(rand.sequences,'pepcats')
#    descr_rand.calculate_crosscorr(7)
#    y_hel = np.zeros(helsamples)
#    y_rand = np.ones(randsamples)
#    data = np.concatenate((descr_hel.descriptor,descr_rand.descriptor), axis=0)
#    labels = np.concatenate((y_hel,y_rand), axis=0)
#    
#    return data,labels
    
    
class DNN(object):
    """
    Base class for constructing a Deep Neural Network
    """
    def __init__(self, archit, activfuncs,num_steps,num_classes):
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
        self.regfactor = 0.02
        self.learningrate = 0.01
        
    def make_graph(self, traindataset, trainlabels, testdataset):
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
                if mode == "train":
                   matrices.append(tf_train_dataset)  # first matrix for the computations is the training data matrix 
                elif mode == "test":
                   matrices.append(tf_test_dataset)  # first matrix for the computations is the test data matrix   
                else:
                    print("invalid computation mode in make_graph")
                    
                for i in range(1,len(self.archit)):    
                    if self.activfuncs[i] == "logits":  # i.e. do nothing -> activation fct. has yet to be applied to this layer.
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
                
#                train_prediction = tf.constant(1,dtype=tf.int32)
#                test_prediction = tf.constant(1,dtype=tf.int32)
                print(matrices[len(self.archit)-1].get_shape())
                print(tf_train_labels.get_shape())
                if mode == "train":
                    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(matrices[len(self.archit)-1], tf_train_labels)) 
                    for i in range(len(self.archit)):  # add error terms for each weightmatrix in the net
                        loss += 0.5 * self.regfactor*tf.nn.l2_loss(weights[i-1])
 #                    optimizer = tf.train.GradientDescentOptimizer(self.learningrate).minimize(loss)
                    optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
                    train_prediction = tf.nn.softmax(matrices[len(self.archit)-1])
                elif mode == "test":
                    test_prediction = tf.nn.softmax(matrices[len(self.archit)-1])
                else:
                    print("invalid computation mode in make_graph(...): Must be either train or test")
                
        return optimizer, train_prediction, test_prediction, graph, loss         
        
    def session(self, graph, optimizer, loss, train_prediction, test_prediction, trainlabels):
        with tf.Session(graph=graph) as session:
            tf.initialize_all_variables().run()
            print("Initialized")
            for step in range(self.num_steps):
                _, l, predictions = session.run([optimizer, loss, train_prediction])
                if step % self.loginterval == 0:
                    print('Loss at step %d: %f' % (step, l))
                    print('Training accuracy: %.1f%%' % accuracy(predictions, trainlabels))
            testprediction = test_prediction.eval()

        return testprediction
        
        
# if __name__ == '__main__':
#    data,labels = gendata()
# #    num_classes = len(np.unique(labels))
#    num_classes = 2
#    traindataset, testdataset, trainlabels, testlabels = \
#    train_test_split(data, labels, test_size=0.33, random_state=42,stratify=labels)
#    # one hot encode labels
#    train_labels = (np.arange(num_classes) == trainlabels[:,None]).astype(np.float32)
#    test_labels = (np.arange(num_classes) == testlabels[:,None]).astype(np.float32)
#    # choose network architecture
#    init_archit = [traindataset.shape[1],10, num_classes]  
#    init_activfuncs = ["input","tanh","logits"]
#    num_steps = 500
#    MyDNN = DNN(init_archit, init_activfuncs, num_steps, num_classes)
#    optimizer, train_prediction, test_prediction, graph, loss = \
#    MyDNN.make_graph(traindataset, train_labels, testdataset)
#    # calculate
#    testprediction = MyDNN.session(graph,optimizer, loss, train_prediction, test_prediction, train_labels)
