from os import listdir
from os.path import isfile, join
from sklearn import svm
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from random import randint
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
import numpy as np
import argparse
from math import *

############################################
#        Access simulation data            #
############################################
def read_training_data(data_dir):

        # Read data 
        signals, delays, shots = read_all_files(data_dir, 0)

        # Convert to np array and matrix for RF
        ground_truths = np.array(delays)
        inputs = np.matrix(signals)

        num_features = inputs.shape[1]
        num_samples = inputs.shape[0]

        return [inputs, ground_truths, num_samples, num_features, shots]

def read_file(filename, row):

        row = row+6

        with open(filename, 'r') as f:
                delay_line = f.readline()
                signal_line = f.readlines()[row].strip().split('\t')
        delay = float(delay_line.strip().split('\t')[1])
        signal = np.array([float(x) for x in signal_line])

        return signal, delay

def read_all_files(dir, row):

        filenames = [f for f in listdir(dir) if isfile(join(dir, f)) and not f.startswith('.')]

        delays = []
        signals = []
        shots = []
        for file in filenames:
                s, d = read_file(dir + file, row)
                signals.append(s)
                delays.append(d)
                shots.append(file[73:])

        return signals, delays, shots

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

def getScores(estimator, x, y):
    yPred = estimator.predict(x)
    return (sqrt(fabs(mean_squared_error(y, yPred))),
            r2_score(y, yPred))

def my_scorer(estimator, x, y):
    rmse, r = getScores(estimator, x, y)
    print rmse, r
    return rmse

def main(args):

	data_dir = args.directory
        folds = args.folds

	print('Reading data...')

	# Access training data -- that is, data in pixel_pos range [350,352)
	[inputs, ground_truths, num_samples, num_features, training_shots] = read_training_data(data_dir)
	
	# Perform cross validation and report mean R^2 value
        param = {'kernel': args.kernel, 'C': args.c, 'gamma': args.gamma}

        # Perform cross validation and report mean R^2 value and RMSE
        print('Performing ' + str(folds) + '-fold cross validation')

	model = svm.SVR(kernel=param['kernel'], 
			C=param['C'], 
			gamma=param['gamma'])

        scores = cross_val_score(model, inputs, ground_truths, cv=folds, scoring=my_scorer)
	print(scores)
        print('RMSE: ' + str(scores.mean()))


if __name__ == "__main__": 

	# Set up arg parser 
        helpstr = 'Perform 3-fold cross validation for support vector machine regressor on simulation data'
        parser = argparse.ArgumentParser(description=helpstr);
	parser.add_argument('-d','--directory',dest='directory', type=str, help='path to directory with original data files', default='../data_simulation/')
	parser.add_argument('-f','--folds',dest='folds', type=int, help='number of folds for cross validation', default=5)
	parser.add_argument('-c', '--c', dest='c', type=float, help='C', default=0.01)
	parser.add_argument('-g', '--gamma', dest='gamma', type=float, help='gamma', default=1000)
	parser.add_argument('-k', '--kernel', dest='kernel', type=str, help='kernel type', default='linear')

        args = parser.parse_args();

        # Call main()
        main(args)


