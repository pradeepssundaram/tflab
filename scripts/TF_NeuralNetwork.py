# bulding a neural network from start
# we have the data so we will write a function to load data from disk
import pandas as pd
import numpy as np
import tensorflow as tf
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
    train_Labels=return1Hot(train_Labels)
    test_Labels=return1Hot(test_Labels)
    return train_Data, train_Labels, test_Data, test_Labels

def prepareANN(X,weights,biases,keep_prob):
    layer_1 = tf.add(tf.matmul(X, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1, keep_prob)
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

# declarations and set up
n_hidden_1 = 40
train_Data, train_Labels, test_Data, test_Labels=preparedatafornn()
n_input = train_Data.shape[1]
n_classes = train_Labels.shape[1]

weights = {
    'h1': tf.Variable(tf.random_normal(shape=[n_input, n_hidden_1],mean=0.,stddev=1/n_input)),
    'out': tf.Variable(tf.random_normal(shape=[n_hidden_1, n_classes],mean=0.,stddev=1/n_input))
}

biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1])),
    'out': tf.Variable(tf.zeros([n_classes]))
}


keep_prob = tf.placeholder("float")

training_epochs = 1000
display_step = 50
batch_size = 200

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

predictions = prepareANN(x, weights, biases, keep_prob)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions,labels=y))

optimizer=tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(training_epochs):
        avg_cost = 0.0
        total_batch = int(len(train_Data) / batch_size)
        x_batches = np.array_split(train_Data, total_batch)
        y_batches = np.array_split(train_Labels, total_batch)
        for i in range(total_batch):
            batch_x, batch_y = x_batches[i], y_batches[i]
            _, c = sess.run([optimizer, cost], 
                            feed_dict={
                                x: batch_x, 
                                y: batch_y, 
                                keep_prob: 0.8
                            })
            avg_cost += c / total_batch
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: test_Data, y: test_Labels, keep_prob: 1.0}))
