import argparse
import randomForestRegressorSim as rf
import supportVectorRegressorSim as svr
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from math import *
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial

def parameter_tune(inputs, truths, ntree=None, mtry=None, depth=None, kernel=None, c=None, gamma=None):
	
	if est == rf:

                param1_name = 'n_estimators'
                param2_name = 'max_features'
                param3_name = 'max_depth'

                # Check if ntree is specified or needs to be tuned
                if ntree is None:
                        param1 = range(900, 1200, 50)
                else:
                        param1 = [ntree]

                # Check if mtry is specified or needs to be tuned
                num_features = inputs[0].shape[1]
                if mtry is None:
                        param2 = [int(floor(sqrt(num_features))),
                                        int(floor(pow(num_features, 0.66667))),
                                        int(floor(pow(num_features, 0.75))),
                                        int(floor(log(num_features, 2))),
                                        int(floor(log(num_features, 4)))]
                else:
                        param2 = [mtry]

		# Check if max depth is specified or needs to be tuned
                if depth is None:
                        param3 = range(10, 55, 5) + [None];
                else:
                        param3 = [depth]

		model = RandomForestRegressor()

        elif est == svr:

                param1_name = 'kernel'
                param2_name = 'C'
                param3_name = 'gamma'

                # Check if kernel is specified or needs to be tuned
                if kernel is None:
                        param1 = ['rbf', 'linear']
                else:
                        param1 = [kernel]

                # Check if C is specified or needs to be tuned
                if c is None:
                        param2 = [0.0001, 0.001, 0.01]
                else:
                        param2 = [c]

		# Check if gamma is specified or needs to be tuned
                if gamma is None:
                        param3 = [1e-2, 1e-3]
                else:
                        param3 = [gamma]

		model = svm.SVR() 

	param_grid = {param1_name: param1, param2_name: param2, param3_name: param3}

	clf = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', n_jobs=10, cv=5)
	clf.fit(inputs, truths)

	print(sqrt(fabs(clf.best_score_)))
	print(clf.best_params_)	


def main(args):

	global est
	if args.model == 'RF':
		est = rf
	elif args.model == 'SVM':
		est = svr

        print('Reading data...')
        [inputs, ground_truths, num_samples, num_features, training_shots] = est.read_training_data(args.directory)

	# Perform parameter tuning
	print('Parameter tuning...')
	if args.model == 'RF':
		parameter_tune(inputs, ground_truths, ntree=args.ntree, mtry=args.mtry, depth=args.depth)
	elif args.model == 'SVM':
		parameter_tune(inputs, ground_truths, kernel=args.kernel, c=args.c, gamma=args.gamma)
	else:
		print('Specified an estimator/model that does not exist')

if __name__ == "__main__":

	# Set up arg parser 
        helpstr = 'Perform parameter tuning for model on data in pixel_pos range [350,352)'
        parser = argparse.ArgumentParser(description=helpstr);
        parser.add_argument('-r','--run', dest='run', type=int, help='run number', default=52)
        parser.add_argument('-d','--directory',dest='directory', type=str, help='path to directory with original data files', 
				default='data_simulation/')
	parser.add_argument('-n','--ntree',dest='ntree',type=int, help='number of trees/estimators in forest', default=None)
	parser.add_argument('-f','--mtry', dest='mtry', type=int, help='number of candidate features at each node split', default=None)
	parser.add_argument('-l','--depth',dest='depth',type=int, help='max depth allowable in tree', default=20)
	parser.add_argument('-k','--kernel',dest='kernel',type=str,help='kernel for mapping', default=None)
	parser.add_argument('-c','--c', dest='c', type=int, help='c value', default=None)
	parser.add_argument('-g','--gamma',dest='gamma',type=float,help='gamma value', default=None)
	parser.add_argument('-m','--model',dest='model',type=str, help='regressor model (RF or SVM)', default='RF')

        args = parser.parse_args()

	main(args)
	
