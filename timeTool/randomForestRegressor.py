from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import *
from random import randint
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
import numpy as np
import argparse
from math import *
import itertools
import operator

############################################
#        Access filtered data 		   #
############################################
def read_training_data(data_dir, write_dir, run_num):
	
	# Read shot numbers to be used for training
	#with open(write_dir + 'xppl3816_r' + str(run_num) + '_training.dat','r') as f:
	#	training_shots = [int(float(line.strip('\n'))) for line in f.readlines()]
	
	# Access delays (ground truths) 
	with open(data_dir + 'xppl3816_r' + str(run_num) + '_delays.dat','r') as f:
		delay_lines = [line for line in f.readlines() if not line.startswith('#')]	
	
	# Access input data
	with open(data_dir + 'xppl3816_r' + str(run_num) + '_matrix.dat','r') as f:
		data_lines = [line for line in f.readlines() if not line.startswith('#')]

	# Filter inputs and delays by training shot numbers
	input_data = []
	delays = []
	for shot in range(len(delay_lines)):
		delays.append(float(delay_lines[shot].strip('\n').split(' ')[2]))
		input_data.append([float(x) for x in data_lines[shot].strip('\n').split(' ')])

	training_shots = range(len(delay_lines))

	# Convert to np array and matrix for RF
	ground_truths = np.array(delays)
	inputs = np.matrix(input_data)
	num_features = inputs.shape[1]
	num_samples = inputs.shape[0]

	return [inputs, ground_truths, num_samples, num_features, training_shots]

############################################
#	Fold data into 3 folds		   #
############################################
def fold_data(inputs, ground_truths, num_features, shots):

	# Keep running sum of how many items get placed in fold 1,2,3 
	zero_sum = 0
        one_sum = 0
        two_sum = 0

	# Track randomly assigned indices
        rand_ints = []
	tracking = {0:[],1:[],2:[]}

	# For every sample, randomly assign to a fold
        for i,shot in enumerate(shots):
                rand_index = randint(0,2)
                tracking[rand_index].append(shot)
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
        for i in range(len(shots)):
                partitioned_data[rand_ints[i][0]][rand_ints[i][1]] = inputs[i]
                partitioned_truths[rand_ints[i][0]][rand_ints[i][1]] = ground_truths[i]

        return [partitioned_data, partitioned_truths, tracking]


############################################
#    Perform 3-fold cross validation       #
############################################ 
def cross_validate(partitioned_data, partitioned_truths, params):
	
	# Keep running total of R^2 values and RMSE
	r2_sum = 0.0
  	rmse_sum = 0.0

	# For each fold
	for k in range(0, 3):
		
		# Testing data is one fold
		testing_data = partitioned_data[k]
		testing_truths = partitioned_truths[k]

		# Training data is remaining two folds
		training_data = np.concatenate((partitioned_data[(k+1)%3], partitioned_data[(k+2)%3]))
		training_truths = np.concatenate((partitioned_truths[(k+1)%3], partitioned_truths[(k+2)%3]))

		# Train and apply forest to train & test data
		params['random_state'] = 2
		forest = train(training_data, training_truths, params)
		y_pred = apply_forest(forest, testing_data)	

		# Score the regression 
		r2_sum += r2_score(testing_truths, y_pred)
		rmse_sum += sqrt(mean_squared_error(testing_truths, y_pred))
	
	# Average over the 3 folds
	return r2_sum/3.0, rmse_sum / 3.0

#########################################
#       Train RF on training_data       #
#########################################
def train(training_data, training_truths, params):

	forest = RandomForestRegressor(n_estimators=params['n_estimators'], 
					max_features=params['max_features'], 
					max_depth=params['max_depth'], 
					random_state=params['random_state'])
	
	forest.fit(training_data, training_truths)

	return forest

#########################################
#       Apply trained RF to test        #
#########################################
def apply_forest(forest, testing_data):

	return forest.predict(testing_data)

#########################################
# Calculcate confidence of predictions  #
#########################################
def calculate_std(model, X):

	pool = Pool()
        std = pool.map(partial(iterate_x, model), X)
        pool.close()
        pool.join()

	return std

def iterate_x(model, x):
 
	preds = []
	for pred in model.estimators_:
		preds.append(pred.predict(x.reshape(1,-1))[0])
	std = np.std(preds)
	
	return std

def main(data_dir, write_dir, run_num):

	# Access training data -- that is, data in pixel_pos range [350,352)
	print('Reading data...')
	[inputs, ground_truths, num_samples, num_features, num_shots] = read_training_data(data_dir, write_dir, run_num)

	# Create the 3 folds by partitioning the data
	print('Folding data...')
	[partitioned_data, partitioned_truths, tracking] = fold_data(inputs, ground_truths, num_features, num_shots)
	
	# Perform cross validation and report mean R^2 value
	print('Cross validation...')
	param = {'n_estimators': 500, 'max_features': 45, 'max_depth': 20}	
	print('3-fold CV R2: ' + str(cross_validate(partitioned_data, partitioned_truths, param)))

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

