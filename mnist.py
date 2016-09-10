#!/usr/bin/python

#https://www.tensorflow.org/versions/r0.7/tutorials/mnist/beginners/index.html

import tensorflow as tf

#read data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

#placeholder to hold the input features
x = tf.placeholder(tf.float32, [None, 784]);

#define the parameters of the regression
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#define the model
y = tf.nn.softmax(tf.matmul(x, W) + b)

#placeholder to hold the input labels
y_ = tf.placeholder(tf.float32, [None, 10]); #for example, if digit 1: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

#define the cost function, cross entropy
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

#training step
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#now, run the computation
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

#learn in batches of 100 input data, 1000 times => 100 000 input data
for i in range(1000):
    if (i + 1) % 100 == 0:
        print("> iteration %d" %(i + 1))
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#now test the model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(b), sess.run(W))
print("accuracy: ", sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}) * 100)
