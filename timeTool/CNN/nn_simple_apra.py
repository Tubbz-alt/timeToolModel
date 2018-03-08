import logging

import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import fully_connected

import argparse

logger = logging.getLogger(__name__)

def simple_preloaded_nn(inputs, labels, keep_prob, alpha, weights):
    # # Probability we dont drop out a neuron
    
    # FC1
    net = fully_connected(
        inputs=inputs, 
        num_outputs=3, 
        weights_initializer=tf.constant_initializer(weights),
        biases_initializer=tf.zeros_initializer(),)

    net = tf.nn.dropout(net, keep_prob)

    # FC2
    net = fully_connected(net, num_outputs=1, activation_fn=None)   

    #Train operations
    with tf.variable_scope("train", reuse=None):
        # RMSE, where net is the outputted predictions
        loss = tf.sqrt(tf.reduce_mean(tf.square(net-labels)))
        # AdamOptimizer is relatively stable and fast as opposed to 
        # GradientDescent
        train_step = tf.train.AdamOptimizer(alpha).minimize(loss)

    return net, loss, train_step

def simple_nn_100(inputs, labels, keep_prob, alpha):
    # # Probability we dont drop out a neuron
    
    # FC1
    net = fully_connected(
        inputs=inputs, 
        num_outputs=100, 
        biases_initializer=tf.zeros_initializer(),
        )

    net = tf.nn.dropout(net, keep_prob)
 
    # FC2
    net = fully_connected(net, num_outputs=1, activation_fn=None)    

    #Train operations
    with tf.variable_scope("train", reuse=None):
        # RMSE, where net is the outputted predictions
        loss = tf.sqrt(tf.reduce_mean(tf.square(net-labels)))        
        # AdamOptimizer is relatively stable and fast as opposed to 
        # GradientDescent
        train_step = tf.train.AdamOptimizer(alpha).minimize(loss)

    return net, loss, train_step

def next_batch(inputs, labels, num, shuffle=False):
    # Perform all the checks when running for the first time
    if not isinstance(num, int):
        raise ValueError('num must be an int. Got {0}'.format(num))
    len_inputs, len_labels = len(inputs), len(labels)
    if len_inputs != len_labels:
        raise ValueError('Inputs and labels must have the same length. Got '
                         '{0} and {1}'.format(len_inputs, len_labels))
    start = 0
    # Grab the next batch every time the generator is called
    while True:
        if shuffle:
            batch_range = np.random.randint(0, len_inputs, num)
        else:
            batch_range = np.arange(start, start+num) % len_inputs
        batch_inputs = inputs[batch_range]
        batch_labels = labels[batch_range]
        yield batch_inputs, batch_labels
        start += num

def load_data():
    print("Loading files...")

    inputs = np.loadtxt('/reg/d/psdm/XPP/xppl3816/scratch/timeTool_ml/data_source/xppl3816_r51_matrix.dat')

    a,b,all_labels = np.loadtxt('/reg/d/psdm/XPP/xppl3816/scratch/timeTool_ml/data_source/xppl3816_r51_delays.dat', unpack=True)

    weights = np.transpose(np.loadtxt('init_weights.txt').reshape(3,1000))

    # Reshape the labels to appease tf
    all_labels = all_labels.reshape(len(all_labels), 1)
    return inputs, a, b, weights, all_labels

# To be used in ipython to help with parameter entry
def run(auto=True):

    print('---------------------------------------')
    print('Fully connected, 2 layer neural network')
    print('---------------------------------------')
    print

    # Will autofill args with defaults
    if auto:
        alpha = 1e-4
        dropout = 1.0
        batchsize = 100
        iterations = 100000
        printn = 20
        network = 3
        print('Running network with the following parameters:')
        print('	alpha 		= {0}'.format(alpha))
        print('	dropout		= {0}'.format(dropout))
        print('	batchsize	= {0}'.format(batchsize))
	print('	iterations	= {0}'.format(iterations))
	print('	printn		= {0}'.format(printn))
	print('	network		= {0}'.format(network)) 

    # Command line interface will walk through entry
    elif not auto:
        alpha = float(raw_input('Enter the learning rate, alpha: ')) 
        dropout = float(raw_input('Enter the keep probability for dropout: ')) 
        batchsize = int(raw_input('Enter the batch size: '))
        iterations = int(raw_input('Enter the number of training iterations: ')) 
        printn = int(raw_input('Enter the number of times loss should be printed in training iterations: ')) 
        network = int(raw_input('Enter the number of neurons desired in hidden layer. Note only 3 or 100 supported: '))

    # Check if user has already loaded data. If not, make sure main() loads data
    r = raw_input('Has the data been pre-loaded? (Y/N): ')
    if r == 'N' or r == 'n':
        initialize_data_variables()

    main(alpha, dropout, batchsize, iterations, printn, network)

def initialize_data_variables():

    global global_inputs, global_loaded_weights, global_labels
    global_inputs, a, b, global_loaded_weights, global_labels = load_data()

def main(alpha_arg, dropout_arg, batchsize_arg, iterations_arg, printn_arg, network_arg):
   
    # Create a standardized set of labels
    all_labels_standardized = (global_labels-global_labels.mean()) / global_labels.std()

    logging.info("Defining placeholder variables")

    # Define the variables
    with tf.variable_scope('input'):
        lineouts = tf.placeholder(tf.float32, shape=[None, 1000])
        labels = tf.placeholder(tf.float32, shape=[None, 1])
        keep_prob = tf.placeholder(tf.float32)
        alpha = tf.placeholder(tf.float32)

    print("Defining networks")
    if(network_arg == 3):
        # Weights pre-initialized
        inp_net, inp_loss, inp_train_step = simple_preloaded_nn(
            lineouts, labels, keep_prob, alpha, global_loaded_weights)
    elif(network_arg==100):
        # Weights set randomly
        inp_net, inp_loss, inp_train_step = simple_nn_100(
            lineouts, labels, keep_prob, alpha)
    
    # Begin the session
    with tf.Session() as sess:
        print("Beginning session")
        sess.run(tf.global_variables_initializer())
    
        # Set the run Parameters
        batch_generator = next_batch(global_inputs, all_labels_standardized, 
                                     batchsize_arg, shuffle=True)

        print("Running for {0} iterations, using a batch size of {1}, "
                    "printing {2} times during training".format(
                        iterations_arg, batchsize_arg, printn_arg))        
        for i in range(iterations_arg):
            # Fetch data of 50 images
            batch = next(batch_generator)

            # Every step run training!
            _, iter_loss, output = sess.run(
                [inp_train_step, inp_loss, inp_net], 
                feed_dict={lineouts: batch[0], labels: batch[1], keep_prob: dropout_arg, alpha: alpha_arg})

            # Print the loss printn times
            if not i % (iterations_arg // printn_arg):
                print("Got a loss of {0} for iteration {1}".format(
                    iter_loss, i))
        
        # Check if converged to case where all values are the same
        if output.mean() == 0. and output.std() == 0.:
            print('Converged incorrectly. Last batch output = {0}'.format(output))

        # Test the network on the same data (for now)
        ret = sess.run(
            [inp_net],
            feed_dict={lineouts: global_inputs, labels: all_labels_standardized, keep_prob: 1.0, alpha: alpha_arg}) 

	# Print unstandardized RMSE
        rmse = np.sqrt(np.mean((((np.multiply(ret,global_labels.std()))+global_labels.mean())-global_labels)**2))
        print("Got an RMSE (unstandardized) of {0}".format(rmse))	
    
if __name__ == "__main__":

    # Set up arg parser 
    helpstr = 'Fully connected, 2 layer neural network'
    parser = argparse.ArgumentParser(description=helpstr);
    parser.add_argument('-a','--alpha', dest='alpha', type=float, help='learning rate', default=1e-4)
    parser.add_argument('-d','--dropout', dest='dropout', type=float, help='keep probability for dropout', default=1.0)
    parser.add_argument('-b','--batchsize', dest='batchsize', type=int, help='batch size', default=100)
    parser.add_argument('-i','--iterations', dest='iterations', type=int, help='num training iterations', default=100000)
    parser.add_argument('-p','--printn', dest='printn', type=int, help='print loss n times throughout iteration', default=20)
    parser.add_argument('-n','--network', dest='network', type=int, help='number of neurons in hidden layer', default=3)

    args = parser.parse_args()

    if not (args.network == 3 or args.network == 100):
        print('{0} neurons in hidden layer not supported'.format(args.network))

    args = parser.parse_args();

    main(args.alpha, args.dropout, args.batchsize, args.iterations, args.printn, args.network)
