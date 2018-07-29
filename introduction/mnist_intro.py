from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

tf.logging.set_verbosity(tf.logging.INFO)



mnist = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)

# Input Layer
input_layer = tf.placeholder(tf.float32, [None, 784])
input_layer1 = tf.reshape(input_layer, [-1, 28, 28, 1])
prediction = tf.placeholder(tf.float32, [None, 10])

# Convolutional Layer #1
# Computes 32 features using a 5x5 filter with ReLU activation.
# Padding is added to preserve width and height.
# Input Tensor Shape: [batch_size, 28, 28, 1]
# Output Tensor Shape: [batch_size, 28, 28, 32]
conv1 = tf.layers.conv2d(
      inputs=input_layer1,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

# Pooling Layer #1
# First max pooling layer with a 2x2 filter and stride of 2
# Input Tensor Shape: [batch_size, 28, 28, 32]
# Output Tensor Shape: [batch_size, 14, 14, 32]
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

# Convolutional Layer #2
# Computes 64 features using a 5x5 filter.
# Padding is added to preserve width and height.
# Input Tensor Shape: [batch_size, 14, 14, 32]
# Output Tensor Shape: [batch_size, 14, 14, 64]
conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
dropout = tf.layers.dropout(
      inputs=dense, rate=0.6)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
logits = tf.layers.dense(inputs=dropout, units=10)

cross_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(prediction, logits))
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cross_entropy)

print("starting training .....")
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
#saver class that we would user to predict out model in the feature
saver=tf.train.Saver()

# Train
for _ in range(100):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={input_layer: batch_xs, prediction: batch_ys})

print("testing")
# Test trained model
testimages=mnist.test.images[:10]
testlabels=mnist.test.labels[:10]
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={input_layer: testimages,
                                      prediction: testlabels}))
saver.save(sess,os.getcwd()+"/model.ckpt")
  
