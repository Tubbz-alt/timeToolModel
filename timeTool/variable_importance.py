import math
import randomForestRegressor 
import linearRegressor
import regression
import argparse
from random import randint
import numpy as np
import extract_row as er
import os
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
from operator import itemgetter

# Read all shots from a run
def access_signals(data_dir, run_num):
	
	with open(data_dir + 'xppl3816_r' + str(run_num) + '_matrix.dat', 'r') as f:
		signals = np.matrix([[float(x) for x in line.strip('\n').split(' ')] for i,line in tqdm(enumerate(f)) if not line.startswith('#')])	

	num_shots = i

	return signals,num_shots
	
# Calculate variable importance for every feature
def calculate_vi(pool, signals, num_shots, original_signal, original_pred, model):

	# Perform multiprocessing
	pred_diff = pool.map(partial(iterate_features, signals, num_shots, original_signal, original_pred, model), range(original_signal.shape[1]))

	# Rank by importance
	temp = np.array(pred_diff).argsort()
	ranks = np.empty(len(pred_diff),int)
	ranks[temp] = np.arange(len(pred_diff))

        return ranks

def iterate_features(signals, num_shots, original_signal, original_pred, model, i):
	
	pred_diff_sum = 0.0

	for j in range(0,3):

        	# Access original signal then replace feature value with randomly selected shot
		permuted_signal = original_signal.copy()
		rand_int = randint(0,num_shots-1)		
		rand_signal = signals[rand_int].copy()
		permuted_signal.put(i, rand_signal.item(i))

      		# Calculate difference in prediction with permutation
		new_pred = model.predict(permuted_signal.reshape(1, -1))
		pred_diff_sum += math.fabs(original_pred - new_pred)

	return pred_diff_sum/3.0

# Write file containing variable importance of each feature to file
def write_vi(write_dir, run_num, shot_num, pred_diff):
	with open(write_dir + 'vi/vi_data/r' + str(run_num) + '_s' + str(shot_num) + '_vi.dat','w') as f:
		f.writelines([str(diff) + '\n' for diff in pred_diff])

# Write gnuplot file to display results overlaid on signal
def write_gnuplot(write_dir, run_num, shot_num):
	with open(write_dir + 'vi/vi.gp','w') as f:
		f.write('set term png\n')
		f.write('set output \'/reg/d/psdm/XPP/xppl3816/scratch/transferLearning/pngs_to_label/' + str(run_num) + '/r' + str(run_num) + '_s' + str(shot_num) + '_vi.png\'\n')
		#f.write('set output \'/reg/d/psdm/XPP/xppl3816/scratch/timeTool_ml/inverted_results/vi/r' + str(run_num) + '_s' + str(shot_num) + '_vi.png\'\n')
		f.write('set xlabel \'index\'\n')
		f.write('set ylabel \'signal\'\n')
		f.write('set cblabel \'importance rank\'\n')
		#f.write('set cbrange [0:1.5]\n')
		f.write('set title \'Variable importance results for run ' + str(run_num) + ', shot ' + str(shot_num) + '\'\n')
		f.write('plot \"< paste \'' + write_dir + 'vi/col_data/r' + str(run_num) + '_s' + str(shot_num) + '_col.dat\' \'' + write_dir + 'vi/vi_data/r' + str(run_num) + '_s' + str(shot_num) + '_vi.dat\'\" u 0:1:2 palette lw 2 with lines title \'opal_0 signal\'\n')
	
	# Run gnuplot script to generate png
	os.system('gnuplot ' + write_dir + 'vi/vi.gp')

def main(train_num, test_num, shot_nums, all_shots, data_dir, write_dir, est):
	
	# Access regressor that was selected
        if est == 'RF':
                estimator_type = randomForestRegressor
        elif est == 'LR':
                estimator_type = linearRegressor
	
	# Access training data -- that is, data in pixel_pos range [350,352)
        print('Reading training data')
	[inputs, x] = regression.access_data(data_dir,write_dir,train_num, 0)
        ground_truths = regression.read_ground_truth(write_dir,train_num, 0)
	
	print('Training model')
	# Train model on training data
	model = estimator_type.train(inputs, ground_truths, 2)

	# Access all shots
	print('Reading testing data')
	signals,num_shots = access_signals(data_dir, test_num)
	
	if all_shots: 
		shot_nums = range(0, num_shots)

	#Prep for multiprocessing
	pool = Pool()

	for shot in tqdm(shot_nums):
	
		shot_num = int(float(shot))

		# Access original signal and perform regression on it
		original_signal = signals[shot_num].copy() 
       		original_pred = model.predict(original_signal.reshape(1, -1))

		# Calculate variable importance for all features
		importances = calculate_vi(pool, signals, num_shots, original_signal, original_pred, model)

		# Write vi results to file
		write_vi(write_dir, test_num, shot_num, importances)

		# Write shot row data to column format for gnuplot
        	er.write_row_to_col(test_num, shot_num, data_dir, write_dir)

		# Write gnuplot file
		write_gnuplot(write_dir, test_num, shot_num)		

	pool.close()
	pool.join()

if __name__=="__main__": 

	# Set up arg parser 
        helpstr = 'Determine most import variable in a particular data point'
        parser = argparse.ArgumentParser(description=helpstr);
        parser.add_argument('-r','--train', dest='train', type=int, help='train run number', default=51)
	parser.add_argument('-t','--test',dest='test',type=int,help='test run number', default=52)
	parser.add_argument('-s','--shots',dest='shots', type=str, help='shot number(s)', default='0')
	parser.add_argument('-a','--all', dest='all_shots', type=bool, help='all shots', default=False)
	parser.add_argument('-d','--directory',dest='directory', type=str, help='path to directory with original data files', default='/reg/d/psdm/XPP/xppl3816/scratch/timeTool_ml/data_source/')
        parser.add_argument('-w','--write',dest='write',type=str, help='path to scratch directory where files are  written to', default='/reg/d/psdm/XPP/xppl3816/scratch/timeTool_ml/data_results/')
	parser.add_argument('-m','--model',dest='model',type=str, help='regression model (RF or LR)', default='RF')

        args = parser.parse_args();

        # Access input
        train_num = args.train
	test_num = args.test
	shot_nums = args.shots.split(' ')
	all_shots = args.all_shots
	data_dir = args.directory
	write_dir = args.write
	est = args.model

        # Call main()
        main(train_num, test_num, shot_nums, all_shots, data_dir, write_dir, est)

