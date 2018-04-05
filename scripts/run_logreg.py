
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 10:24:41 2018

@author: psundara
"""

import sys

sys.path.append("D:\\BecomingADS\\TflabPyCharm\\tflab")

import numpy as np
import tensorflow as tf
import sklearn
from sklearn.datasets import fetch_mldata

from network import FeedForwardSMRegression, FeedForwardRegression
from optimizers import ASGradientDescentOptimizer, ASRMSPropOptimizer


def PrepareData():
    custom_data_home = "Data"
    mnist=fetch_mldata('mnist-original', data_home="D:\BecomingADS\TflabPyCharm\Data")
    train_x_l = mnist.data
    train_x_l = train_x_l/255
    train_y_l = mnist.target
    n_samples_l = mnist.data.shape[0]
    n_features_l = mnist.data.shape[1]
    n_classes_l = len(np.unique(mnist.target))
    train_y_l = train_y_l.astype(np.int16)
    train_y_l = np.eye(n_classes_l)[train_y_l]
    return train_x_l, train_y_l, n_samples_l, n_features_l, n_classes_l
    

# Parameters
steps = 200
learning_rate = 0.001
train_x,train_y,n_samples,n_features,n_classes = PrepareData()

rng = np.random
rng.seed(1234)

# Training Data


opts = [
    tf.train.GradientDescentOptimizer(learning_rate=learning_rate),
    ASGradientDescentOptimizer(base_learning_rate=learning_rate,scale=1.0001),
    # tf.train.RMSPropOptimizer(learning_rate=learning_rate),
    # ASRMSPropOptimizer(base_learning_rate=learning_rate,scale=1.0001),
    # tf.train.AdamOptimizer(learning_rate=learning_rate),
    # tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=.9),
    # tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=.9, use_nesterov=True),
    # tf.train.AdagradOptimizer(learning_rate=learning_rate)
]
opt_names = [
        'SGD',
        'SGD+AS',
        # 'RMSProp',
        # 'RMSProp+AS',
        # 'ADAM',
        # 'SGD+M',
        # 'SGD+NM',
        # 'Adagrad'
        ]

# Launch the graph
losses = []

for i, opt in enumerate(opts):

    with tf.Session() as sess:
        print(opt_names[i])
        reg = FeedForwardSMRegression([n_features, n_classes], nonlinearities=lambda x: tf.exp(x)/tf.reduce_sum(tf.exp(x)))
        loss = reg.train(sess, train_x, train_y, minibatch_size=200,
                         steps=steps, optimizer=opts[i],optimizer_name=opt_names[i])
        losses.append(loss)
    tf.reset_default_graph()
