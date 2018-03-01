# Import tensorflow
import tensorflow as tf

# Import xavier_initializer, a type of initializer we will use for weights
from tensorflow.contrib.layers import xavier_initializer

# Fully connected layer (every pixel maps to every neuron)
def fully_connected(input_tensor, name, num_output, init_weights=None, activation=None):
    # Flatten the input tensor to 1D array 
    shape=input_tensor.get_shape()
    flat_size = 1
    for index in xrange(len(shape)-1):
        flat_size *= shape[1-len(shape)+index].value # flat_size = ending_w * ending_h * ending_channels
    input_tensor = tf.reshape(input_tensor, [-1,flat_size]) # -1 means tensorflow, go figure out what this number should be

    # Create graph nodes within this layer's scope
    with tf.variable_scope(name):
        # Weights for neurons
	if init_weights is None:
	    weights = tf.get_variable(name='%s_weights' % name,
                                  shape=[input_tensor.get_shape()[-1].value, num_output],
                                  dtype=tf.float32,
                                  initializer=xavier_initializer())
	else:
	    weights = tf.get_variable(name='%s_weights' % name,
                                  shape=[input_tensor.get_shape()[-1].value, num_output],
                                  dtype=tf.float32,
                                  initializer=tf.constant(init_weights))

        # Biases for neurons
        biases  = tf.get_variable(name='%s_biases' % name,
                                  shape=[num_output],
                                  dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.1))
        
        # Compute neuron sums
        result = tf.matmul(input_tensor,weights) + biases

        # Activation
        if activation:
            result = activation(result)

        # Register variables to be monitored in tensorboard
        tf.summary.histogram('%s_weights' % name, weights)
        tf.summary.histogram('%s_biases' % name, biases)

    return result
  
# Network inputs
with tf.variable_scope('input'):
    images    = tf.placeholder(tf.float32, shape=[None, 784])
    labels    = tf.placeholder(tf.float32, shape=[None, 10])
    keep_prob = tf.placeholder(tf.float32)
    
    image2d   = tf.reshape(images, [-1, 28, 28, 1])
    # Record a random 10 samples of images in monitoring
    tf.summary.image('images',image2d,10)

init_weights = np.loadtxt('init_weights.txt')
 
# FC1
net = fully_connected(net, 'fc1', 1024, init_weights=init_weights, activation=tf.nn.relu)
with tf.variable_scope('drop_out'):      # Drop out: a technique to avoid overfitting
                                         # disconnect certain connections to neurons from image to image 
    net = tf.nn.dropout(net, keep_prob)
# FC2
net = fully_connected(net, 'fc2', 10)    # 10 classes, so 10 output neurons

# Train operations
with tf.variable_scope("train"):
  
    # This is NOT a probability! This is softmax (or softmax probability is OK too) 
    # and represents a distribution forced to be in [0., 1.] range
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=net)) # logits = network prediction
                                                                                              # labels = correct answers
      
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss) # AdamOptimizer is relatively stable and fast (as opposed to GradientDescent)
    
    correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(labels, 1)) # Compare prediction labels
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))     # Fraction of images that were correctly predicted
    
    # Monitor these loss and accuracy during training
    tf.summary.scalar('loss',loss)
    tf.summary.scalar('accuracy',accuracy)
    
# Define a session
sess = tf.InteractiveSession()
# Create log monitor
import os
if not os.path.isdir('tb_log'): os.makedirs('tb_log')
log_writer = tf.summary.FileWriter('tb_log')
log_writer.add_graph(sess.graph)
summary_op = tf.summary.merge_all()

# Let's time this
import time
start = time.time()

# Ready! initialize and train for 5000 steps
sess.run(tf.global_variables_initializer())
for i in range(2000):
    batch = mnist.train.next_batch(50)
    
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={images: batch[0], labels: batch[1], keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))

    if i % 20 == 0:
        s = sess.run(summary_op, feed_dict={images: batch[0], labels: batch[1], keep_prob: 1.0})
        log_writer.add_summary(s,i)
        
    train_step.run(feed_dict={images: batch[0], labels: batch[1], keep_prob: 0.5})
    
print('test accuracy %g' % accuracy.eval(feed_dict={images: mnist.test.images, labels: mnist.test.labels, keep_prob: 1.0}))
print(time.time() - start)

# Define operations: a softmax probability, and the prediction by the network
with tf.variable_scope('analysis'):
    softmax = tf.nn.softmax(logits=net)
    prediction_label = tf.argmax(net,1)

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#%matplotlib inline
# Dump some output images
#prob_array, pred_label, = sess.run([softmax,prediction_label],feed_dict={images: mnist.test.images,keep_prob:1.0})
#for index in xrange(10):
#    print 'Prediction\033[91m',pred_label[index],'\033[00mwith softmax prob\033[94m',prob_array[index][pred_label[index]],'\033[00m'
#    plt.imshow(mnist.test.images[index].reshape([28,28]).astype(np.float32),cmap='gray',interpolation='none')
#    plt.axis('off')
##    plt.show()
