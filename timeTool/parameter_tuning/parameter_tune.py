import argparse
import randomForestRegressor as rf
import supportVectorRegressor as svr
from math import *
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial

def parameter_tune(folded_data, folded_truths, ntree=None, mtry=None, depth=None, kernel=None, c=None, gamma=None):

	if est == rf:
	
		param1_name = 'n_estimators'
		param2_name = 'max_features'
		param3_name = 'max_depth'

		# Check if ntree is specified or needs to be tuned
		if ntree is None:
			param1 = range(950, 2000, 50)
		else:
			param1 = [ntree]
	
		# Check if mtry is specified or needs to be tuned
        	num_features = folded_data[0].shape[1]
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
			param2 = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
		else:
			param2 = [c]
	
		# Check if gamma is specified or needs to be tuned
		if gamma is None:
			param3 = [1e-2, 1e-3, 1e-4, 1e-5]
		else:
			param3 = [gamma]	


	# Generate all possible combinations
	param_combinations = []
	for a in param1:
		for b in param2:
			if param1_name == 'kernel' and param1[0] == 'linear':
				param = {param1_name: a, 
					param2_name: b}
				param_combinations.append(param)	
			else:
				for c in param3:
					param = {param1_name: a, 
						param2_name: b,
						param3_name:c}
					param_combinations.append(param)
 
	pool = Pool()
	all_scores = pool.map(partial(iterate_params, folded_data, folded_truths), param_combinations)
	pool.close()
	pool.join() 

	best_score = (1,100000)
	best_params = {}
	print('Complete results:')
        print(str(param1_name) + '\t' + str(param2_name) + '\t' + str(param3_name) + '\t|\tR^2\t\t\tRMSE')
        print('------------------------------------------------------------------------------------------------')
        for index,(r2_score, rmse_score) in enumerate(all_scores):
                params = param_combinations[index]
                print(str(params[param1_name]) + '\t\t'
                        + str(params[param2_name]) + '\t\t'
                        + str(params[param3_name]) + '\t\t' + '|' + '\t' 
			+ str(r2_score) + '\t\t'
			+ str(rmse_score))

		if rmse_score < best_score[1]:
			best_score = (r2_score, rmse_score)
			best_params[param1_name] = params[param1_name]
			best_params[param2_name] = params[param2_name]
			best_params[param3_name] = params[param3_name]

	print('------------------------------------------------------------------------------------------------')
	print
	print('Best results: R2=' + str(best_score[0]) + ' RMSE=' + str(best_score[1]))
	print(str(param1_name) + '=' + str(best_params[param1_name]))
	print(str(param2_name) + '=' + str(best_params[param2_name]))
	print(str(param3_name) + '=' + str(best_params[param3_name]))

def iterate_params(folded_data, folded_truths, params):

	r2, rmse = est.cross_validate(folded_data, folded_truths, params)

        return r2, rmse
	

def main(args):

	global est
	if args.model == 'RF':
		est = rf
	elif args.model == 'SVM':
		est = svr

        print('Reading data...')
        [inputs, ground_truths, num_samples, num_features, training_shots] = est.read_training_data(args.directory, args.write, args.run)

        # Create the 3 folds by partitioning the data
        print('Folding data...')
        [partitioned_data, partitioned_truths, tracking] = est.fold_data(inputs, ground_truths, num_features, training_shots)
		
	# Perform parameter tuning
	print('Parameter tuning...')
	if args.model == 'RF':
		parameter_tune(partitioned_data, partitioned_truths, ntree=args.ntree, mtry=args.mtry, depth=args.depth)
	elif args.model == 'SVM':
		parameter_tune(partitioned_data, partitioned_truths, kernel=args.kernel, c=args.c, gamma=args.gamma)
	else:
		print('Specified an estimator/model that does not exist')

if __name__ == "__main__":

	# Set up arg parser 
        helpstr = 'Perform parameter tuning for model on data in pixel_pos range [350,352)'
        parser = argparse.ArgumentParser(description=helpstr);
        parser.add_argument('-r','--run', dest='run', type=int, help='run number', default=52)
        parser.add_argument('-d','--directory',dest='directory', type=str, help='path to directory with original data files', 
				default='/reg/d/psdm/XPP/xppl3816/scratch/timeTool_ml/data_source/')
        parser.add_argument('-w','--write',dest='write',type=str, help='path to directory to write files to', 
				default='/reg/d/psdm/XPP/xppl3816/scratch/timeTool_ml/data_results/')
	parser.add_argument('-n','--ntree',dest='ntree',type=int, help='number of trees/estimators in forest', default=None)
	parser.add_argument('-f','--mtry', dest='mtry', type=int, help='number of candidate features at each node split', default=None)
	parser.add_argument('-l','--depth',dest='depth',type=int, help='max depth allowable in tree', default=None)
	parser.add_argument('-k','--kernel',dest='kernel',type=str,help='kernel for mapping', default=None)
	parser.add_argument('-c','--c', dest='c', type=int, help='c value', default=None)
	parser.add_argument('-g','--gamma',dest='gamma',type=float,help='gamma value', default=None)
	parser.add_argument('-m','--model',dest='model',type=str, help='regressor model (RF or SVM)', default='RF')

        args = parser.parse_args()

	main(args)
	
