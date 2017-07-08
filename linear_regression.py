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
# use a dynamic uncertainty of 5% and an independent uncertainty of 0.1
y_train_meas = [np.random.normal(0.,0.05*abs(y_train[i]))+np.random.normal(0.,0.1)+y_train[i] for i in range(len(y_train))]

# nominal training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(500):# do 1000 iterations of the gradient descent to numcerically minimize the loss function
        sess.run(train, {x:x_train, y:y_train_meas})
        if i%100==0:
            curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train_meas})
            print("Training step: %s W: %s b: %s loss: %s"%(i,curr_W, curr_b, curr_loss))
W_, b_, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train_meas})
print("--------------------------------------------------------------------------------")
print("final values of parameters for nominal measurement obtained by training/optimization")
print("W: %s b: %s loss: %s"%(W_, b_, curr_loss))
print("--------------------------------------------------------------------------------")

# now throw 100 toys with the given uncertainty on the y values to determine the uncertainty on the fitted parameters W and b
Ws=np.array([])
bs=np.array([])
print("--------------------------------------------------------------------------------------------------------------")
print "starting to throw toys with the given uncertainty on the y values and doing a linear regression for each toy"
print("--------------------------------------------------------------------------------------------------------------")
for k in range(200):
    #reset the initial values of W and b for every toy
    sess.run(init)
    #add the random gaussian error to the nominal y values
    y_train_mc= [np.random.normal(0.,0.05*abs(y_train[i]))+np.random.normal(0.,0.1)+y_train[i] for i in range(len(y_train))]
    for i in range(500):# do 1000 iterations of the gradient descent to numcerically minimize the loss function
        sess.run(train, {x:x_train, y:y_train_mc})
        if i%100==0:
            curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train_mc})
            #print("Training step: %s W: %s b: %s loss: %s"%(i,curr_W, curr_b, curr_loss))
    # evaluate training accuracy of toys
    print("")
    print("final values of parameters for toy %s"%k)
    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x:x_train, y:y_train_meas})
    Ws=np.append(Ws,curr_W)
    bs=np.append(bs,curr_b)
    print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
    print("")
    
# standard deviations of the distribution of Ws and bs obtained from the toys    
sigma_W=np.std(Ws)
sigma_b=np.std(bs)
print("standard deviation of the Ws obtained by throwing toys: %s"%sigma_W)
print("standard deviation of the bs obtained by throwing toys: %s"%sigma_b)
# correlation matrix of the fitted Ws and bs obtained from the toy generation
print("correlation matrix of the regressed Ws and bs:")
correlation = np.corrcoef(Ws,bs)
print correlation

# plot the nominal measurement with error bars and the regressed linear model including the uncertainty on the fitted parameters
#plt.plot(x_train,y_train_meas,'ro')
plt.errorbar(x_train,y_train_meas,xerr=0.,yerr=[0.05*abs(y_train[i])+0.1 for i in range(len(y_train))],fmt='ko')
plt.plot(x_train,[W_*x_+b_ for x_ in x_train],'-')
plt.plot(x_train,[(W_+sigma_W)*x_+(b_+correlation[0,1]*sigma_b) for x_ in x_train],'--')
plt.plot(x_train,[(W_-sigma_W)*x_+(b_-correlation[0,1]*sigma_b) for x_ in x_train],'--')
plt.plot(x_train,[(W_+correlation[0,1]*sigma_W)*x_+(b_+sigma_b) for x_ in x_train],'--')
plt.plot(x_train,[(W_-correlation[0,1]*sigma_W)*x_+(b_-sigma_b) for x_ in x_train],'--')

plt.ylabel('y')
plt.xlabel('x')
plt.title('Linear regression with y=W*x+b')
plt.grid(True)
plt.savefig("linear_regression.pdf")
plt.show()

plt.scatter(Ws,bs)
plt.ylabel('Ws')
plt.xlabel('bs')
plt.title('2D distribution of fitted Ws and bs')
plt.grid(True)
plt.savefig("linear_regression_correlation.pdf")
plt.show()
