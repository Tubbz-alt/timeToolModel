"""
Reimplementation of simple nn
"""
import logging

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

from timeTool.CNN.base import BaseModel, Dataset, BaseTrainer
from timeTool.CNN.tfutils import lazy_property, define_scope

logger = logging.getLogger(__name__)


class SimpleModel(BaseModel):
    """Simple three neuron model."""
    @lazy_property
    @define_scope
    def model(self):
        # FC1
        net = fully_connected(
            inputs=self.inputs, 
            num_outputs=3, 
            weights_initializer=tf.constant_initializer(
                self.data.weights.as_matrix()),
            biases_initializer=tf.zeros_initializer()
        )

        net = tf.nn.dropout(net, self.keep_prob)

        # FC2
        net = fully_connected(net, num_outputs=1, activation_fn=None)
        return net
    
    @lazy_property
    def loss(self):
        # RMSE, where net is the outputted predictions
        loss = tf.sqrt(tf.reduce_mean(tf.square(self.model-self.labels)))
        return loss

    @lazy_property
    @define_scope
    def optimizer(self):
        # GradientDescent
        train = tf.train.AdamOptimizer(self.alpha).minimize(self.loss)
        return train


class SimpleDataset(Dataset):
    def __init__(self, *args, weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights = weights if isinstance(weights, pd.DataFrame) \
                       else pd.DataFrame(weights)


class SimpleTrainer(BaseTrainer):
    def train(self, model, epochs=100):
        # Initialize the variables
        self.initialize()

        # Perform the training loop
        for inputs, labels in self.data.epochs(epochs):
            _, loss, output = self.sess.run(
                [model.optimizer, model.loss, model.model],
                feed_dict={model.inputs: inputs, model.labels: labels,
                           model.keep_prob: 1.0, model.alpha: 3e-5})

        # Check if converged to case where all values are the same
        if output.mean() == 0. and output.std() == 0.:
            print('Converged incorrectly. Last batch output = {0}'.format(
                output))

        # Test the network on the same data (for now)
        loss, ret = self.sess.run([model.loss, model.model], feed_dict={
            model.inputs: self.data.x, model.labels: self.data.y,
            model.keep_prob: 1.0, model.alpha: 3e-5})

        print('Got an average loss of: {0}'.format(loss))
