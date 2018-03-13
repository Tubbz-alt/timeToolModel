"""
Base classes for tensorflow models
"""
from abc import ABC, abstractmethod
import logging

import numpy as np
import pandas as pd
import tensorflow as tf

from timeTool.CNN.tfutils import lazy_property, define_scope

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Base model that organizes all tensorflow models."""
    def __init__(self, name=None, sess=None, config=None, data=None):
        self.name = name
        self.sess = sess
        self.config = config
        self.data = data

        self._default_data_shape = [None, 1]

        # Initialize the placeholders
        self.init_placeholders()

        # # Build out the models
        # logger.debug("Instantiating model...")
        # self.model
        # logger.debug("Instantiating loss...")
        # self.loss
        # logger.debug("Instantiating train...")
        # self.train

    @lazy_property
    @define_scope
    def model(self):
        pass

    @lazy_property
    @define_scope
    def loss(self):
        pass
    
    @lazy_property
    @define_scope
    def train(self):
        pass

    @define_scope
    def init_placeholders(self):
        # Variables that will always be needed
        self.keep_prob = tf.placeholder(tf.float32)
        self.alpha = tf.placeholder(tf.float32)        

        # If a data object was passed in, use that to determine the x and y
        # placeholders
        if self.data is not None:
            self.x = tf.placeholder(tf.float32, shape=self.data.model_x_shape)
            self.y = tf.placeholder(tf.float32, shape=self.data.model_y_shape)
        # Otherwise initialize them to be the default data shape
        else:
            self.x = tf.placeholder(tf.float32, shape=self._default_data_shape)
            self.y = tf.placeholder(tf.float32, shape=self._default_data_shape)
        
    def save(self, sess=None, checkpoint_dir=None, global_step_tensor=None):
        """save the checkpoint in the path defined in the config file."""
        self.saver.save(
            sess or self.sess,
            checkpoint_dir or self.config.checkpoint_dir,
            global_step_tensor or self.global_step_tensor)

    def load(self, sess):
        """Load latest checkpoint from the experiment path."""
        latest_checkpoint = tf.train.latest_checkpoint(
            self.config.checkpoint_dir)
        if latest_checkpoint:
            self.saver.restore(sess or self.sess, latest_checkpoint)
        
    def __call__(self, *args, **kwargs):
        self.model(*args, **kwargs)


class Dataset:
    """Abstract dataset class"""
    def __init__(self, x, y, batch_size=1, shuffle=False, loop=False):
        # Turn the inputs into a dataframe
        self.x = x if isinstance(x, pd.DataFrame) else pd.DataFrame(x)
        self.y = x if isinstance(y, pd.DataFrame) else pd.DataFrame(y)

        # Ensure the length of both x and y are the same
        if len(self.x) != len(self.y):
            raise ValueError("Inputted X and y have differing lengths, {0} and "
                             "{1}.".format(len(self.x), len(self.y)))

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.loop = loop
        self.n_samples = 0
        self.epochs = 0

        self._end_epoch = False
        self._remaining_indices = np.arange(len(self))

    @property
    def model_x_shape(self):
        return [None, *self.x.shape[1:]]

    @property
    def model_y_shape(self):
        return [None, *self.y.shape[1:]]

    def __len__(self):
        return len(self.y)
    
    @property
    def _batch_indices(self):
        # If shuffle is true, return a random set of indices that is either the
        # batch size in length or the length of the remaining indices
        if self.shuffle:
            selection_indices = np.radom.choice(
                len(self._remaining_indices),
                size=min(self.batch_size, len(self._remaining_indices)),
                replace=False)
        # Otherwise just create a range of the batch size or the remaining
        # indices length
        else:
            selection_indices = np.arange(
                min(self.batch_size, len(self._remaining_indices)))

        # Grab the real indices that will be used to sample the data
        return_indices = self._remaining_indices[selection_indices]
        # Increment the internal number samples counter
        self.n_samples += len(return_indices)

        # If we have more remaining indices than selection indices, then
        # remove the selection indices for the nect iteration
        if len(self._remaining_indices) > len(selection_indices):
            self._remaining_indices = np.delete(self._remaining_indices,
                                                selection_indices)
            
        else:
            # Otherwise, replenish the remaining indices
            self._remaining_indices = np.arange(len(self))
            # Signal that this was the end of the epoch
            self._end_epoch = True
            self.epochs += 1

        # Finally return the indices to grab the data by
        return return_indices
        
    def __iter__(self):
        return self

    def __next__(self):
        # End the iteration if we do not want to loop and we have exceeded the
        # length of the dataset
        if not self.loop and self._end_epoch:
            self._end_epoch = False
            raise StopIteration
        # Grab the batch indices to use for this iteration
        indices = self._batch_indices
        # Return the x and y values indexed using the batch indices
        return self.x.iloc[indices,:], self.y.iloc[indices,:]
