import tensorflow as tf
import numpy
pandas as pd
import matplotlib.pyplot as plt

rng = numpy.random

spreadsheet = 'LR_ML.xlsx'
data = pd.read_excel(spreedsheet)

months = data['Machine Age (Months)'].values
MTBF = data['Mean Time Between Failure (Days)'].values

#hyperparameters
learning_rate = 0.01
training_epochs = 1000


#paramater
display_step = 50


train_X = numpy.asarray(months)
train_Y = numpy.asarray(MTBF)


n_namples = train_X.shape[0]

X = tf.placeholder('float')
Y = tf.placeholder('float')

W = tf.Variable(rng.randn(),name="Weight")
b = tf.Variable(rng.randn(), name = 'bias')

pred = tf.add(tf.multiply(X,W),b)

error = tf.reduce_sum(tf.pow(pred-Y,2))/(2*n_samples)

optimizer = tf.train.GradientDescentOptimixer(lean_rate).minimize(error)

init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(error, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "error=", "{:.9f}".format(c), \
                "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_error = sess.run(error, feed_dict={X: train_X, Y: train_Y})
    print("Training error=", training_error, "W=", sess.run(W), "b=", sess.run(b), '\n')

    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()

    # Testing example, as requested (Issue #2)
    test_X = numpy.asarray([2,4,6,8,10])
    test_Y = numpy.asarray([25,23,21,19,17])

    print("Testing... (Mean square loss Comparison)")
    testing_error = sess.run(
        tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * test_X.shape[0]),
        feed_dict={X: test_X, Y: test_Y})  # same function as cost above
    print("Testing error=", testing_error)
    print("Absolute mean square loss difference:", abs(
        training_error - testing_error))

    plt.plot(test_X, test_Y, 'bo', label='Testing data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
