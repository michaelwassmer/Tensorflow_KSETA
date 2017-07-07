import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Model parameters for a linear model y = W*x+b
W = tf.Variable([-2.], dtype=tf.float32)# weight/slope of the model as variable because it has to be modied during runtime/regression
b = tf.Variable([2.], dtype=tf.float32)# bias of the model as variable because it has to be modified during runtime/regression
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
x_train = [1.,2.,3.,4.,5.,6.]
y_train = [0.,-1.,-2.,-3.,-4.,-5.]
y_train_meas = np.random.normal(0.,0.2,len(y_train))+y_train

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong

Ws=np.array([])
bs=np.array([])
print Ws
for k in range(100):
    sess.run(init)
    y_train_mc= np.random.normal(0.,0.2,len(y_train))+y_train
    for i in range(1000):# do 1000 iterations of the gradient descent to numcerically minimize the loss function
        sess.run(train, {x:x_train, y:y_train_mc})
        if i%100==0:
            curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train_mc})
            print("Training step: %s W: %s b: %s loss: %s"%(i,curr_W, curr_b, curr_loss))
            
    # evaluate training accuracy
    print("--------------------------------------------------------------------------------")
    print("final values of parameters")
    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train_meas})
    Ws=np.append(Ws,curr_W)
    bs=np.append(bs,curr_b)
    print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
    print("--------------------------------------------------------------------------------")
    
sigma_W=np.std(Ws)
sigma_b=np.std(bs)
print Ws
print bs
print sigma_W
print sigma_b
correlation = np.corrcoef(Ws,bs)

print correlation
print correlation[0,1]
plt.plot(x_train,y_train_meas,'ro')

W_ = float(Ws[0])
b_ = float(bs[0])
plt.plot(x_train,[W_*x_+b_ for x_ in x_train],'-')
plt.plot(x_train,[(W_+sigma_W)*x_+(b_+correlation[0,1]*sigma_b) for x_ in x_train],'-')
plt.plot(x_train,[(W_-sigma_W)*x_+(b_-correlation[0,1]*sigma_b) for x_ in x_train],'-')
plt.plot(x_train,[(W_+correlation[0,1]*sigma_W)*x_+(b_+sigma_b) for x_ in x_train],'-')
plt.plot(x_train,[(W_-correlation[0,1]*sigma_W)*x_+(b_-sigma_b) for x_ in x_train],'-')

plt.ylabel('y')
plt.xlabel('x')
plt.title('Linear regression with y=W*x+b')
plt.show()
