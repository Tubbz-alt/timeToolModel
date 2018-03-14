"""
Base classes for tensorflow models
"""
from abc import ABC, abstractmethod
from collections import namedtuple
from copy import copy
from itertools import repeat
import logging

import numpy as np
import pandas as pd
import tensorflow as tf

from timeTool.CNN.tfutils import lazy_property, define_scope

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for tensorflow models.

    Being an ABC (Abstract Base Class), it requires that it is inherited (rather
    than instantiated directly) and three high-level methods be overridden:
    ``model``, ``loss``, and ``optimizer``, each aptly named for their intended
    use. Model is expected to contain the architechture of the intended model,
    returning the resulting tensor, loss implements the loss function to be
    used by the training sequence, and optimizer implements the training
    sequence.

    When these methods are implemented, make sure to add the ``lazy_property``
    decorator to them. In addition to making the method a property, it will make
    it so the defined operations are not added to the computational graph every
    time the property is called.

    Parameters
    ----------
    name : str, optional
    	A name for the model instance

    sess : tf.Session, optional
    	This is just so the session can be accessed from within the object if
    	needed

    data : Dataset, optional
    	Dataset object with the data that is being used

    x : Container, optional
    	Data structure containing the input data
    
    y : Container, optional
    	Data structure containing the label data
    """
    def __init__(self, name=None, sess=None, data=None, x=None, y=None):
        self.name = str(name)
        self.sess = sess

        # If x and y were inputted and not data, create a dataset object
        if data is None and x is not None and y is not None:
            self.data = Dataset(x, y)
        # Otherwise just use whatever was provided in the data argument
        else:
            self.data = data

        # Define a default data shape for the input and label placeholders
        self._default_data_shape = [None, 1]

        # Initialize the placeholders
        self.init_placeholders()
        # Initialize the full model architecture
        self.model
        self.loss
        self.optimizer

    @lazy_property
    @abstractmethod
    def model(self):
        """Architecture of the model"""
        pass

    @lazy_property
    @abstractmethod
    def loss(self):
        """Loss function of the model"""
        pass
    
    @lazy_property
    @abstractmethod
    def optimizer(self):
        """Training sequence of the model"""
        pass

    @define_scope
    def init_placeholders(self):
        """Method to initialize all the placeholder variables.

        While the previous three methods are meant to be overridden, this method
        is expetected to be extended to include any new placeholders necessary
        for the model to run. It currently instantiates the keep probablity for
        dropout, learning rate alpha, the inputs (depending on the dataset), and
        labels (depending on the dataset).
        """
        # Variables that will always be needed
        self.keep_prob = tf.placeholder(tf.float32)
        self.alpha = tf.placeholder(tf.float32)        

        # If a data object was passed in, use that to determine the x and y
        # placeholders
        if self.data is not None:
            self.inputs = tf.placeholder(tf.float32,
                                         shape=self.data.model_x_shape)
            self.labels = tf.placeholder(tf.float32,
                                         shape=self.data.model_y_shape)
        # Otherwise initialize them to be the default data shape
        else:
            self.inputs = tf.placeholder(tf.float32,
                                         shape=self._default_data_shape)
            self.labels = tf.placeholder(tf.float32,
                                         shape=self._default_data_shape)
        
    def save(self, sess=None, checkpoint_dir=None, global_step_tensor=None):
        """Save the checkpoint in the path defined in the config file."""
        raise NotImplementedError('This is kept here for future integration, '
                                  'but has not been tested.')
        self.saver.save(
            sess or self.sess,
            checkpoint_dir or self.config.checkpoint_dir,
            global_step_tensor or self.global_step_tensor)

    def load(self, sess):
        """Load latest checkpoint from the experiment path."""
        raise NotImplementedError('This is kept here for future integration, '
                                  'but has not been tested.')
        latest_checkpoint = tf.train.latest_checkpoint(
            self.config.checkpoint_dir)
        if latest_checkpoint:
            self.saver.restore(sess or self.sess, latest_checkpoint)


class Dataset:
    """Dataset class to wrap the inputs and labels into something singular.

    This class simply guarantees a uniform interface between the data and all
    models that will use it. It makes checks to ensure the data is well formed
    and then definess some attributes for keeping track of the state.

    Besides providing some meta-data to the models, the primary purpose for this
    class is to make the data an iterator to grab batches. This means that it
    can be made into a for loop like so:

    	In [1]: mydataset = Dataset(inputs, labels)
    	   ...: for x, y in mydataset:
    	   ...:     ...

    In this exmaple, X and y contain corresponding batches defined the batch
    size parameter for the object.

    Additionally, true epochs are supported by the iterations. This means that
    ragardless of whether the data is set to be shuffled or not, batch inputs
    are selected without replacement, until all of the data has been returned.
    Once the all the data has been yielded, the internel attribute ``epoch`` is
    incremented, and then looping through the dataset will simply begin anew.

    ..note: This means that the last batch in the iteration will be of a
    		batch-size unless len(data) % batch_size is zero

    Parameters
    ----------
    x : Container
    	Input data in the form of an array or dataframe. It will be converted to
    	a dataframe when it is instantiated

    y : Container
    	Labels for the data in the form of an array or dataframe. It will be
    	converted to a dataframe when instantiated

    batch_size : int, optional
    	Batch size for data batches

    shuffle : bool, optional
    	Shuffle the data when returning batches

    loop : bool, optional
    	Continue iterating indefinitely when used in a loop. This still results
    	in a potentially different batch size for the last iteration in the
    	epoch. It is expected that if this is set to ``True``, the loop will be
    	interrupted by a ``break``.

    replacement : bool, optional
    	If shuffle is ``True``, select datapoints for the batch with replacement

    Attributes
    ----------
    n_samples : int
    	Number of samples that have been returned in total thus far

    n_epochs : int
    	Number of completed epochs

    n_batches : int
    	Number of batches that have been returned in total thus far
    """
    def __init__(self, x, y, batch_size=1, shuffle=False, loop=False,
                 replacement=False):
        # Turn the inputs into a dataframe
        self.x = x if isinstance(x, pd.DataFrame) else pd.DataFrame(x)
        self.y = x if isinstance(y, pd.DataFrame) else pd.DataFrame(y)

        # Ensure the length of both x and y are the same
        if len(self.x) != len(self.y):
            raise ValueError("Inputted X and y have differing lengths, {0} and "
                             "{1}.".format(len(self.x), len(self.y)))

        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.loop = loop
        self.replacement = replacement

        # Useful counters
        self._n_samples = 0
        self._n_epochs = 0
        self._n_batches = 0

        # Internal state indicators and modifiers
        self._end_epoch = False
        self._remaining_indices = np.arange(len(self))
        self._batch = namedtuple('batch', ['x','y'])

    @property
    def batches_per_epoch(self):
        return int(np.ceil(len(self) / self.batch_size))

    def reset(self):
        """Resets the iterator to the start of the dataset."""
        self._remaining_indices = np.arange(len(self))

    @property
    def remaining(self):
        """Returns the number of remaining samples in this epoch."""
        return len(self._remaining_indices)
        
    @property
    def n_epochs(self):
        """Returns the number of epochs that have completed."""
        return self._n_epochs

    @property
    def n_samples(self):
        """Returns the number of samples that have been returned."""
        return self._n_samples

    @property
    def n_batches(self):
        """Returns the number of batches that have been returned."""
        return self._n_batches
        
    @property
    def model_x_shape(self):
        """Returns the same shape of the input data but with a None as the first
        element.
        """
        return [None, *self.x.shape[1:]]

    @property
    def model_y_shape(self):
        """Returns the same shape of the input data but with a None as the first
        element.
        """
        return [None, *self.y.shape[1:]]

    def __len__(self):
        """Implements the ``len`` operator for all ``Dataset`` instances.

        This simply means if you call ``len(mydataset)``, it will return the
        same thing as ``len(mydataset.y)``.
        """
        return len(self.y)
    
    @property
    def _batch_indices(self):
        """Internal method that selects the next set of indices return in a
        batch.
        """
        # If shuffle is true, return a random set of indices that is either the
        # batch size in length or the length of the remaining indices
        if self.shuffle:
            # If sampling with replacement, reset the indices every time
            if replacement:
                self.reset()
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
        # Increment the internal number of samples and batches
        self._n_samples += len(return_indices)
        self._n_batches += 1

        # If we have more remaining indices than selection indices, then
        # remove the selection indices for the next iteration
        if len(self._remaining_indices) > len(selection_indices):
            self._remaining_indices = np.delete(self._remaining_indices,
                                                selection_indices)
            
        else:
            # Otherwise, replenish the remaining indices
            self.reset()
            # Signal that this was the end of the epoch
            self._end_epoch = True
            self._n_epochs += 1

        # Finally return the indices to grab the data by
        return return_indices
        
    def __iter__(self):
        """Make this class an iterator."""
        return self

    def __next__(self):
        """Action to perform at every iteration in a loop (invocation of
        next()).
        """
        # End the iteration if we do not want to loop and we have exceeded the
        # length of the dataset
        if not self.loop and self._end_epoch:
            self._end_epoch = False
            raise StopIteration
        
        # Grab the batch indices to use for this iteration
        indices = self._batch_indices
        # Return the x and y values indexed using the batch indices as matrices
        return self._batch(self.x.iloc[indices,:].as_matrix(),
                           self.y.iloc[indices,:].as_matrix())

    def epochs(self, num_epochs):
        """Returns an iterator that is set to stop after the inputted number of
        epochs.

        The primary intended use of this method is to allow the user to loop
        through batches until the desired number of epochs has been completed.
        Since this class has been implemented as an iterator, it can be used
        as follows:

    		In [1]: mydataset = Dataset(inputs, labels)
    	   	   ...: for x, y in mydataset.epochs(10):
    	       ...:     ....

        In this example, the for loop will continue until the 10 epochs have
        been completed.

        ..note: This will reset the current position of the iterator to the
        		start of the dataset.

        Parameters
        ----------
        num_epochs : int
        	Number of epochs to iterate through
        """
        # Reset the remaining indices of this instance
        self.reset()
        # Figure out the number batches necessary to achieve the desired number
        # of epochs
        num_batches = self.batches_per_epoch * num_epochs
        
        # Return the iterator that runs next for the inputted number of batches
        def epoch_generator():
            # Backup the current loop parameter
            loop_backup = self.loop
            # Make sure the loop will continue as long as we want
            self.loop = True
            # Loop through all the required
            for i in range(num_batches):
                yield next(self)
            # Set the loop back to what it was initially
            self.loop = loop_backup
        return epoch_generator()
        
    def batches(self, num_batches):
        """Returns an iterator that is set to stop after the inputted number of
        batches.

        The primary intended use of this method is to allow the user to loop
        through batches until the desired number of batches have been returned.
        Since this class has been implemented as an iterator, it can be used
        as follows:

    		In [1]: mydataset = Dataset(inputs, labels)
    	   	   ...: for x, y in mydataset.batches(10):
    	       ...:     ....

        In this example, the for loop will continue until the 10 batches have
        been returned.

        Parameters
        ----------
        num_batches : int
        	Number of batches to return
        """
        def batch_generator():
            # Backup the current loop parameter
            loop_backup = self.loop
            # Make sure the loop will continue as long as we want
            self.loop = True
            # Loop through all the desired batches
            for i in range(num_batches):
                yield next(self)
            # Set the loop back to what it was initially
            self.loop = loop_backup
        return batch_generator()
    
class BaseTrainer(ABC):
    """Trainer class that will be used to train models.

    Being an ABC (Abstract Base Class), this class needs to be inherited (rather
    then used directly) and only ``train`` needs to be overridden. This methon
    should implement the actual training sequence to use on the model. It is
    also recommended that all changes needed to be made to the tensorflow
    session be made here, such as instantiating summary operations.

    Additionally there are several convenience properties that return properties
    of the inputted dataset.
    """
    def __init__(self, sess, data):
        self.sess = sess
        self.data = data
        self.init = tf.group(tf.global_variables_initializer(),
                             tf.local_variables_initializer())
        
    @abstractmethod
    def train(self):
        pass

    def initialize(self):
        self.sess.run(self.init)

    @property
    def epochs(self):
        return self.data.n_epochs

    @property
    def samples(self):
        return self.data.n_samples

    @property
    def batches(self):
        return self.data.n_batches

    @property
    def batch_size(self):
        return self.data.batch_size

    @batch_size.setter
    def batch_size(self, size):
        self.data.batch_size = int(size)
