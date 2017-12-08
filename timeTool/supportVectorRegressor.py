from sklearn import svm
from sklearn.metrics import r2_score
from random import randint
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
import numpy as np
import argparse
import math

############################################
#        Access filtered data 		   #
############################################
def read_training_data(data_dir, write_dir, run_num):
	
	# Read shot numbers to be used for training
	with open(write_dir + 'xppl3816_r' + str(run_num) + '_training.dat','r') as f:
		training_shots = [int(float(line.strip('\n'))) for line in f.readlines()]
	
	# Access delays (ground truths) 
	with open(data_dir + 'xppl3816_r' + str(run_num) + '_delays.dat','r') as f:
		delay_lines = [line for line in f.readlines() if not line.startswith('#')]	
	
	# Access input data
	with open(data_dir + 'xppl3816_r' + str(run_num) + '_matrix.dat','r') as f:
		data_lines = [line for line in f.readlines() if not line.startswith('#')]

	# Filter inputs and delays by training shot numbers
	input_data = []
	delays = []
	for shot in training_shots:
		delays.append(float(delay_lines[shot].strip('\n').split(' ')[1]))
		input_data.append([float(x) for x in data_lines[shot].strip('\n').split(' ')])

	# Convert to np array and matrix for RF
	ground_truths = np.array(delays)
	inputs = np.matrix(input_data)
	num_features = inputs.shape[1]
	num_samples = inputs.shape[0]

	return [inputs, ground_truths, num_samples, num_features]

############################################
#	Fold data into 3 folds		   #
############################################
def fold_data(inputs, ground_truths, num_samples, num_features):

	# Keep running sum of how many items get placed in fold 1,2,3 
	zero_sum = 0
        one_sum = 0
        two_sum = 0

	# Track randomly assigned indices
        rand_ints = []

	# For every sample, randomly assign to a fold
        for i in range(0, num_samples):
                rand_index = randint(0,2)
                if rand_index == 0:
                        rand_ints.append([rand_index, zero_sum])
                        zero_sum += 1
                elif rand_index == 1:
                        rand_ints.append([rand_index, one_sum])
                        one_sum += 1
                else:
                        rand_ints.append([rand_index, two_sum])
                        two_sum += 1
        partition_sums = [zero_sum, one_sum, two_sum]

	# Build empty matrix and array for data and truth
        partitioned_data = [np.zeros([partition_sum, num_features]) for partition_sum in partition_sums]
        partitioned_truths = [np.zeros([partition_sum]) for partition_sum in partition_sums]
	
	# Fill matrix and array with appropriate data and truth according to rand_ints
        for i in range(0, num_samples):
                partitioned_data[rand_ints[i][0]][rand_ints[i][1]] = inputs[i]
                partitioned_truths[rand_ints[i][0]][rand_ints[i][1]] = ground_truths[i]

        return [partitioned_data, partitioned_truths]


############################################
#    Perform 3-fold cross validation       #
############################################ 
def cross_validate(partitioned_data, partitioned_truths, params):
	
	# Keep running total of R^2 values
	r2_sum = 0.0

	# For each fold
	for k in range(0, 3):
		
		# Testing data is one fold
		testing_data = partitioned_data[k]
		testing_truths = partitioned_truths[k]

		# Training data is remaining two folds
		training_data = np.concatenate((partitioned_data[(k+1)%3], partitioned_data[(k+2)%3]))
		training_truths = np.concatenate((partitioned_truths[(k+1)%3], partitioned_truths[(k+2)%3]))

		# Train and apply SVM to train & test data
		model = train(training_data, training_truths, params)
		y_pred = apply_model(model, testing_data)	

		# Score the regression 
		r2_sum += r2_score(testing_truths, y_pred)
	
	# Average over the 3 folds
	return r2_sum / 3.0

#########################################
#       Train SVM on training_data       #
#########################################
def train(training_data, training_truths, params):

	if 'kernel' in params:
		kernel = params['kernel']
	else:
		kernel = 'rbf'
	if 'C' in params:
		C = params['C']
	else:
		C = 1.0
	if kernel == 'rbf':
		if 'gamma' in params:
			gamma = params['gamma']
		else:
			gamma = 1e-3
		model = svm.SVR(kernel=kernel, C=C, gamma=gamma)
	
	elif kernel == 'linear':
		model = svm.SVR(kernel=kernel, C=C)

	model.fit(training_data, training_truths)

	return model

#########################################
#       Apply trained SVM to test       #
#########################################
def apply_model(model, testing_data):

	return model.predict(testing_data)

#########################################
# Calculcate confidence of predictions  #
#########################################
#def calculcate_ci(model, X, percentile=95):

#	pool = Pool()
#        ci = pool.map(partial(iterate_x, model, percentile), X)
#        pool.close()
#        pool.join()

#	return ci

#def iterate_x(model, percentile, x):
 
	# Somehow measure variance?
	
#	err_down = np.percentile(preds, (100 - percentile) / 2.)
#	err_up = np.percentile(preds, 100 - (100 - percentile) / 2.)
	
#	return (err_down, err_up)

def main(data_dir, write_dir, run_num):

	print('Reading data...')

	# Access training data -- that is, data in pixel_pos range [350,352)
	[inputs, ground_truths, num_samples, num_features] = read_training_data(data_dir, write_dir, run_num)

	# Create the 3 folds by partitioning the data
	[partitioned_data, partitioned_truths] = fold_data(inputs, ground_truths, num_samples, num_features)
	
	# Perform cross validation and report mean R^2 value
        print('3-fold CV R2: ' + str(cross_validate(partitioned_data, partitioned_truths)))

if __name__ == "__main__": 

	# Set up arg parser 
        helpstr = 'Perform 3-fold cross validation for random forest regressor on data in pixel_pos range [350,352)'
        parser = argparse.ArgumentParser(description=helpstr);
        parser.add_argument('-r','--run', dest='run', type=int, help='run number', default=52)
	parser.add_argument('-d','--directory',dest='directory', type=str, help='path to directory with original data files', default='/reg/d/psdm/XPP/xppl3816/scratch/timeTool_ml/data_source/')
        parser.add_argument('-w','--write',dest='write',type=str, help='path to directory to write files to', default='/reg/d/psdm/XPP/xppl3816/scratch/timeTool_ml/data_results/')

        args = parser.parse_args();

        # Access input
        run_num = args.run
	data_dir = args.directory
	write_dir = args.write

        # Call main()
        main(data_dir, write_dir, run_num)

