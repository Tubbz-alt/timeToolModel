import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import fully_connected

inputs = np.loadtxt('/reg/d/psdm/XPP/xppl3816/scratch/timeTool_ml/data_source/xppl3816_r51_matrix.dat')

a,b,labels = np.loadtxt('/reg/d/psdm/XPP/xppl3816/scratch/timeTool_ml/data_source/xppl3816_r51_delays.dat', unpack=True)

weights = np.transpose(np.loadtxt('init_weights.txt').reshape(3,1023))

# FC1
net = fully_connected(inputs=inputs, weights_initializer=tf.constant(weights))

with tf.variable_scope('drop_out'):      # Drop out: a technique to avoid overfitting
                                         # disconnect certain connections to neurons from image to image 
    net = tf.nn.dropout(net, keep_prob)

# FC2
net = fully_connected(net, num_outputs=1, activation_fn=None)

# Train operations
#with tf.variable_scope("train"):

    # This is NOT a probability! This is softmax (or softmax probability is OK too) 
    # and represents a distribution forced to be in [0., 1.] range
#    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=net)) # logits = network prediction
                                                                                              # labels = correct answers

#    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss) # AdamOptimizer is relatively stable and fast (as opposed to GradientDescent)

#    correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(labels, 1)) # Compare prediction labels
#    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))     # Fraction of images that were correctly predicted

    # Monitor these loss and accuracy during training
#    tf.summary.scalar('loss',loss)
#    tf.summary.scalar('accuracy',accuracy)

# Define a session
#sess = tf.InteractiveSession()


