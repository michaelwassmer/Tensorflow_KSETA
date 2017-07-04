# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.examples.tutorials.mnist import input_data

# Import data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import tensorflow as tf

sess = tf.InteractiveSession()

# Create the model

# a 2d tensor for the input batches: for each image 784 pixels and the number of training objects in the batch is arbitrary, therefore the keyword "None"
x = tf.placeholder(tf.float32, [None, 784])

# the weights in shape of a 784x10 matrix and the biases for each output class
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# the linear regression model as a matrix multiplication of the weight matrix with the input vector and the final addition of the biases to each output node
y = tf.matmul(x, W) + b

# another 2d tensor for the target output
y_ = tf.placeholder(tf.float32, [None, 10])

tf.global_variables_initializer().run()

# the loss function is chosen to be the cross entropy between the target and the models prediction
cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))

# use the steepest gradient descent algorithm to minimize the loss function
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Test trained model
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Train
for _ in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
	if _%100==0:
            curr_loss = sess.run(cross_entropy, {x:mnist.test.images, y_:mnist.test.labels})
            curr_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
            print("Training step: %s loss: %s accuracy: %s"%(_, curr_loss,curr_acc))


print("Accuracy of the model is after 1000 batch trainings: ",sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

