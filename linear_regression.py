import numpy as np
import tensorflow as tf

# Model parameters for a linear model y = W*x+b
W = tf.Variable([.3], dtype=tf.float32)# weight/slope of the model as variable because it has to be modied during runtime/regression
b = tf.Variable([-.3], dtype=tf.float32)# bias of the model as variable because it has to be modified during runtime/regression
# Model input and output
x = tf.placeholder(tf.float32)# input variable as placeholder since it has to be given for the regression
linear_model = W * x + b
y = tf.placeholder(tf.float32)# target value as placeholder since it has to be given for the regression
# loss
loss = tf.reduce_sum(tf.square(linear_model - y))# sum of the squares as loss function to quantify the difference between the true value and the predicted/regressed value
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)# initialize a minimizer 
train = optimizer.minimize(loss)# minimize the loss function by modifying the slope and bias with a minimization algorithm, for example steepest gradient descent
# training data following y=-1*x+1
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):# do 1000 iterations of the gradient descent to numcerically minimize the loss function
    sess.run(train, {x:x_train, y:y_train})
    if i%100==0:
        curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
        print("Training step: %s W: %s b: %s loss: %s"%(i,curr_W, curr_b, curr_loss))

# evaluate training accuracy
print("--------------------------------------------------------------------------------")
print("final values of parameters")
print("--------------------------------------------------------------------------------")
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
