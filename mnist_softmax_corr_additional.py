# example for MNIST with TensorFlow
# adapted from TensorFlow MNIST Tutorial on TensorFlow Webpage and from https://www.oreilly.com/learning/not-another-mnist-tutorial-with-tensorflow

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import random as ran
import tensorflow as tf

# Import data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True,validation_size=15000)


# function to return the tensor containing the training images (x_train) and the tensor containing the training labels (y_label)
def TRAIN_SIZE(num):
    print ('Total Training Images in Dataset = ' + str(mnist.train.images.shape))
    print ('--------------------------------------------------')
    x_train = mnist.train.images[:num,:]
    print ('x_train Examples Loaded = ' + str(x_train.shape))
    y_train = mnist.train.labels[:num,:]
    print ('y_train Examples Loaded = ' + str(y_train.shape))
    print('')
    return x_train, y_train
# function to return the tensor containing the testing images (x_train) and the tensor containing the testing labels (y_label)
def TEST_SIZE(num):
    print ('Total Test Examples in Dataset = ' + str(mnist.test.images.shape))
    print ('--------------------------------------------------')
    x_test = mnist.test.images[:num,:]
    print ('x_test Examples Loaded = ' + str(x_test.shape))
    y_test = mnist.test.labels[:num,:]
    print ('y_test Examples Loaded = ' + str(y_test.shape))
    print('')
    return x_test, y_test
# function to display a random image of the training data
def display_digit(num,x_train,y_train):
    print(y_train[num])
    label = y_train[num].argmax(axis=0)
    image = x_train[num].reshape([28,28])
    plt.title('Example: %d  Label: %d' % (num, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()
# function to flatten the training images and show the image vectors corresponding to the images 
def display_mult_flat(start, stop, x_train):
    images = x_train[start].reshape([1,784])
    for i in range(start+1,stop):
        images = np.concatenate((images, x_train[i].reshape([1,784])))
    plt.imshow(images, cmap=plt.get_cmap('gray_r'))
    plt.title("flattened form of image %d to %d"%(start,stop))
    plt.show()
# display a random image and compare the predicted to the true value
def display_compare(num):
    # THIS WILL LOAD ONE TRAINING EXAMPLE
    x_train = mnist.test.images[num,:].reshape(1,784)
    y_train = mnist.test.labels[num,:]
    # THIS GETS OUR LABEL AS A INTEGER
    label = y_train.argmax()
    # THIS GETS OUR PREDICTION AS A INTEGER
    prediction = sess.run(y, feed_dict={x: x_train}).argmax()
    plt.title('Prediction: %d Label: %d' % (prediction, label))
    plt.imshow(x_train.reshape([28,28]), cmap=plt.get_cmap('gray_r'))
    plt.savefig("mnist_exmaple_prediction.pdf")
    plt.show()

sess = tf.InteractiveSession()

# Create the model
# a 2d tensor for the input batches: for each image 784 pixels and the number of training objects in the batch is arbitrary, therefore the keyword "None"
x = tf.placeholder(tf.float32, [None, 784])
# the weights in shape of a 784x10 matrix and the biases for each output class
W = tf.Variable(tf.random_normal([784, 800]))
W_neu = tf.Variable(tf.random_normal([800,10]))
b = tf.Variable(tf.random_normal([800]))
b_neu = tf.Variable(tf.random_normal([10]))
# the linear regression model as a matrix multiplication of the weight matrix with the input vector and the final addition of the biases to each output node
y = tf.matmul(tf.nn.sigmoid(tf.matmul(x, W) + b),W_neu)+b_neu
# another 2d tensor for the target output
y_ = tf.placeholder(tf.float32, [None, 10])


# load training and test samples
x_train, y_train = TRAIN_SIZE(45000)
x_test, y_test = TEST_SIZE(10000)

# the loss function is chosen to be the cross entropy between the target and the models prediction
cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
# learning rate of minimization rate for the GradientDescentOptimizer
learning_rate = 0.005
# use the steepest gradient descent algorithm to minimize the loss function
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# Test trained model
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# display random image
print("displaying random image now")
print("one hot vector of random image")
display_digit(ran.randint(0, x_train.shape[0]),x_train,y_train)

# flattened form of the images
display_mult_flat(0,500,x_train)

# set the number of training epochs and the batch size for the single trainings
training_epochs=500
batch_size=150
tf.global_variables_initializer().run()
# Train
#loss_test=[]
loss_train=[]
loss_val=[]
#acc_test=[]
acc_train=[]
acc_val=[]
epoch=[]
for _ in range(training_epochs):
        #do batch learning
	batch_xs, batch_ys = mnist.train.next_batch(batch_size)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
	if _%10==0:
            epoch.append(_)
            #curr_acc,curr_loss = sess.run([accuracy,cross_entropy], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
            #loss_test.append(curr_loss)
            #acc_test.append(curr_acc)
            #print("Training step: %s loss: %s accuracy: %s (test sample)"%(_, curr_loss,curr_acc))
            curr_acc,curr_loss = sess.run([accuracy,cross_entropy], feed_dict={x: mnist.train.images, y_: mnist.train.labels})
            loss_train.append(curr_loss)
            acc_train.append(curr_acc)
            print("Training step: %s loss: %s accuracy: %s (train sample)"%(_, curr_loss,curr_acc))
            curr_acc,curr_loss = sess.run([accuracy,cross_entropy], feed_dict={x: mnist.validation.images, y_: mnist.validation.labels})
            loss_val.append(curr_loss)
            acc_val.append(curr_acc)
            print("Training step: %s loss: %s accuracy: %s (val sample)"%(_, curr_loss,curr_acc))


# plot loss function for test and training sample
#plt.plot(epoch,loss_test,'r',label='loss: test sample')
plt.plot(epoch,loss_train,'b',label='loss: training sample')
plt.plot(epoch,loss_val,'g',label='loss: validation sample')
#plt.plot(epoch,acc_test,'--r',label='accuracy: test sample')
plt.plot(epoch,acc_train,'--b',label='accuracy: training sample')
plt.plot(epoch,acc_val,'--g',label='accuracy: validation sample')
plt.legend()
plt.title("loss and accuracy")
plt.xlabel('training epochs')
plt.ylabel('loss function/accuracy')
plt.savefig("mnist_loss_and_accuracy.pdf")
plt.show()

print("##################################################################################################################")
print("Accuracy of the model after %d batch trainings with a batch size of %d is %s "%(training_epochs,batch_size,sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})))
print("##################################################################################################################")

for i in range(10):
    plt.subplot(2, 5, i+1)
    weight = sess.run(W)[:,i]
    plt.title(i)
    plt.imshow(weight.reshape([28,28]), cmap=plt.get_cmap('seismic'))
    plt.colorbar()
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
print("showing the weights corresponding to the different possible digits")
plt.savefig("mnist_weights.pdf")
plt.show()

display_compare(ran.randint(0, 10000))



