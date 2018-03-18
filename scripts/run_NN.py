#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 23:38:37 2018

@author: Pradeep Sundaram
"""


import pandas as pd
import numpy as np
import tensorflow as tf
import sys
import matplotlib.pyplot as plt
#sys.append('../tflab/tflab/')
sys.path.append('/media/admin1/60221789221762F8/tflab/tflab/')
from network import FeedForwardSMRegression,FeedForwardRegression,FeedForwardANN
from optimizers import ASGradientDescentOptimizer, ASRMSPropOptimizer

def getdatafromdisk(trainfile, testfile):

    train_raw=pd.read_csv(trainfile)
    test_raw=pd.read_csv(testfile)
    train_x = train_raw.iloc[:, 1:]
    train_y = train_raw.iloc[:, 0]
    test_x = test_raw.iloc[:, 1:]
    test_y = test_raw.iloc[:, 0]

    return train_x.values, train_y.values.astype(np.int8)\
        , test_x.values, test_y.values.astype(np.int8)

def return1Hot(data):
    n_classes=len(np.unique(data))
    retval = np.eye(n_classes)[data]
    return retval

def preparedatafornn():
    # one hot encoding of labels
    train_Data, train_Labels, test_Data, test_Labels = \
        getdatafromdisk('../data/mnist_train.csv','../data/mnist_test.csv')
    num_classes=len(np.unique(train_Labels))
    train_Labels=return1Hot(train_Labels)
    test_Labels=return1Hot(test_Labels)
    return train_Data, train_Labels, test_Data, test_Labels,num_classes

#def prepareANN(X,weights,biases,keep_prob):
#    layer_1 = tf.add(tf.matmul(X, weights['h1']), biases['b1'])
#    layer_1 = tf.nn.relu(layer_1)
#    layer_1 = tf.nn.dropout(layer_1, keep_prob)
#    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
#    return out_layer

# Parameters
import time
start_time=time.clock()
steps = 5000
learning_rate = 0.001
train_Data, train_Labels, test_Data, test_Labels,num_classes=preparedatafornn()
num_features=train_Data.shape[1]

num_hidden=40
rng = np.random
rng.seed(1234)

# Training Data


opts = [
    tf.train.GradientDescentOptimizer(learning_rate=learning_rate),
    ASGradientDescentOptimizer(base_learning_rate=learning_rate,scale=1.0001),
    tf.train.RMSPropOptimizer(learning_rate=learning_rate),
    ASRMSPropOptimizer(base_learning_rate=learning_rate,scale=1.0001),
    tf.train.AdamOptimizer(learning_rate=learning_rate),
    tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=.9),
    tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=.9, use_nesterov=True),
    tf.train.AdagradOptimizer(learning_rate=learning_rate)
]
opt_names = [
        'SGD', 
        'SGD+AS', 
        'RMSProp', 
        'RMSProp+AS', 
        'ADAM', 
        'SGD+M', 
        'SGD+NM', 
        'Adagrad'
        ]

# Launch the graph
losses = []
with tf.Session() as sess:
    for i, opt in enumerate(opts):
        print(opt_names[i])
        reg = FeedForwardANN([num_features, num_hidden,num_classes], nonlinearities=[tf.nn.relu,tf.nn.softmax])
        loss = reg.train(sess, train_Data, train_Labels, minibatch_size=200,
                         steps=steps, optimizer=opts[i])
        losses.append(loss)

#plt.clf()
#for loss, opt_name in zip(losses, opt_names):
#    plt.plot(loss[::1000], '+-', alpha=.5, label=opt_name)
#plt.legend()
#plt.show()
#
print('Total Time Taken = :')
print(time.clock() - start_time)