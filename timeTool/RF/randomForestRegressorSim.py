from os import listdir
from os.path import isfile, join
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import *
from sklearn.model_selection import cross_val_score
from random import randint
from multiprocessing import Pool
#from tqdm import tqdm
from functools import partial
import numpy as np
import argparse
from math import *
import itertools
import operator

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

	print(str(num_features) + ' FEATURES')

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

        # Access training data -- that is, data in pixel_pos range [350,352)
        print('Reading data...')
        [inputs, ground_truths, num_samples, num_features, shots] = read_training_data(data_dir)

	param = {'n_estimators': args.ntree, 'max_depth': args.depth, 'random_state': 2}
 	if args.mtry == 'sqrt' or args.mtry == '1/2':
		param['max_features'] = int(sqrt(num_features))
	elif args.mtry == '2/3':
		param['max_features'] = int(num_features**(2./3))
	elif args.mtry == '3/4':
		param['max_features'] = int(num_features**(3./4))
	elif args.mtry == 'log2':
		param['max_features'] = int(log(num_features, 2))
		print(param['max_features'])
	elif args.mtry == 'log4':
		param['max_features'] = int(log(num_features, 4))
	else:
		print('Invalid mtry entry. Options are 1/2 or sqrt, 2/3, 3/4, log2, or log4.')

        # Perform cross validation and report mean R^2 value and RMSE
        print('Performing ' + str(folds) + '-fold cross validation')
	
	forest = RandomForestRegressor(n_estimators=param['n_estimators'],
                                        max_features=param['max_features'],
                                        max_depth=param['max_depth'],
                                        random_state=param['random_state'])
	
	scores = cross_val_score(forest, inputs, ground_truths, cv=folds, scoring=my_scorer)
	print(scores)
	print('RMSE: ' + str(scores.mean()))

if __name__ == "__main__":

        # Set up arg parser 
        helpstr = 'Perform 3-fold cross validation for random forest regressor on simulation data'
        parser = argparse.ArgumentParser(description=helpstr);
        parser.add_argument('-d','--directory',dest='directory', type=str, help='path to directory with original data files', default='data_simulation/')
	parser.add_argument('-f', '--folds', dest='folds', type=int, help='number of folds for cross validation', default=5)
	parser.add_argument('-n', '--ntree', dest='ntree', type=int, help='number of trees in forest', default=165)
  	parser.add_argument('-m', '--mtry', dest='mtry', type=str, help='number of candidate features at each node split', default='sqrt')
	parser.add_argument('-l', '--depth', dest='depth', type=int, help='max tree depth', default=20)

        args = parser.parse_args();

        # Call main()
        main(args)

