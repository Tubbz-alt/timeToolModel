import math
import randomForestRegressor 
import linearRegressor
import argparse
from random import randint
import numpy as np
import extract_row as er
import os
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial

# Read all shots from a run
def access_signals():
	global signals, num_shots
	
	with open(data_dir + 'xppl3816_r' + str(run_num) + '_matrix.dat', 'r') as f:
		signals = np.matrix([[float(x) for x in line.strip('\n').split(' ')] for i,line in tqdm(enumerate(f)) if not line.startswith('#')])	

	num_shots = i
	
# Calculate variable importance for every feature
def calculate_vi(original_signal, original_pred):

	# Prep for multiprocessing
	pool = Pool()

	pred_diff = pool.map(partial(iterate_features, original_signal, original_pred), range(original_signal.shape[1]))

	pool.close()

        return pred_diff

def iterate_features(original_signal, original_pred, i):
	
	pred_diff_sum = 0.0

        # Permute 3 times for each feature and average for more precise answer
        for j in range(0,3):

        	# Access original signal then replace feature value with randomly selected shot
		permuted_signal = original_signal
		rand_signal = signals[randint(0,num_shots-1)]
		permuted_signal.put(i, rand_signal.item(i))

                # Calculate difference in prediction with permutation
		new_pred = model.predict(permuted_signal.reshape(1, -1))
		pred_diff_sum += math.fabs(original_pred - new_pred)

                # Average 3 instances
		pred_d = pred_diff_sum / 3.0

	return pred_d

# Write file containing variable importance of each feature to file
def write_vi(write_dir, shot_num, pred_diff):
	with open(write_dir + 'vi/vi_data/r' + str(run_num) + '_s' + str(shot_num) + '_vi.dat','w') as f:
		f.writelines([str(diff) + '\n' for diff in pred_diff])

# Write gnuplot file to display results overlaid on signal
def write_gnuplot(write_dir, shot_num):
	with open(write_dir + 'vi/vi.gp','w') as f:
		f.write('set term png\n')
		f.write('set output \'/reg/d/psdm/XPP/xppl3816/scratch/transferLearning/pngs_to_label/r' + str(run_num) + '_s' + str(shot_num) + '_vi.png\'\n')
		f.write('set xlabel \'index\'\n')
		f.write('set ylabel \'signal\'\n')
		f.write('set cblabel \'prediction error\'\n')
		f.write('set cbrange [0:2.5]\n')
		f.write('set title \'Variable importance results for run ' + str(run_num) + ', shot ' + str(shot_num) + '\'\n')
		f.write('plot \"< paste \'' + write_dir + 'vi/col_data/r' + str(run_num) + '_s' + str(shot_num) + '_col.dat\' \'' + write_dir + 'vi/vi_data/r' + str(run_num) + '_s' + str(shot_num) + '_vi.dat\'\" u 0:1:2 palette lw 2 with lines title \'opal_0 signal\'\n')
	
	# Run gnuplot script to generate png
	os.system('gnuplot ' + write_dir + 'vi/vi.gp')

def main(run, shot_nums, all_shots, d_dir, write_dir, est):
	global data_dir, run_num, model
	
	data_dir = d_dir
	run_num = run
	
	# Access regressor that was selected
        if est == 'RF':
                estimator_type = randomForestRegressor
        elif est == 'LR':
                estimator_type = linearRegressor
	
	# Access training data -- that is, data in pixel_pos range [350,352)
        [inputs, ground_truths, num_samples, num_features] = estimator_type.read_training_data(data_dir, write_dir, run_num)
	
	# Train forest on training data
	model = estimator_type.train(inputs, ground_truths)

	# Access all shots
	print('Reading data file')
	access_signals()
	
	# Loop through all specified shot numbers
	pool = Pool()
	
	print('Calculating variable importances')
	if all_shots: 
		shot_nums = range(0, num_shots)

	for shot in tqdm(shot_nums):
	
		shot_num = int(float(shot))

		# Access original signal and perform regression on it
		original_signal = signals[shot_num] 
       		original_pred = model.predict(original_signal.reshape(1, -1))

		# Calculate variable importance for all features
		importances = calculate_vi(original_signal, original_pred)

		# Write vi results to file
		write_vi(write_dir, shot_num, importances)

		# Write shot row data to column format for gnuplot
        	er.write_row_to_col(run_num, shot_num, data_dir, write_dir)

		# Write gnuplot file
		write_gnuplot(write_dir, shot_num)		

if __name__=="__main__": 

	# Set up arg parser 
        helpstr = 'Determine most import variable in a particular data point'
        parser = argparse.ArgumentParser(description=helpstr);
        parser.add_argument('-r','--run', dest='run', type=int, help='run number', default=52)
	parser.add_argument('-s','--shots',dest='shots', type=str, help='shot number(s)', default='0')
	parser.add_argument('-a','--all', dest='all_shots', type=bool, help='all shots', default=False)
	parser.add_argument('-d','--directory',dest='directory', type=str, help='path to directory with original data files', default='/reg/d/psdm/XPP/xppl3816/scratch/timeTool_ml/data_source/')
        parser.add_argument('-w','--write',dest='write',type=str, help='path to scratch directory where files are  written to', default='/reg/d/psdm/XPP/xppl3816/scratch/timeTool_ml/data_results/')
	parser.add_argument('-m','--model',dest='model',type=str, help='regression model (RF or LR)', default='RF')

        args = parser.parse_args();

        # Access input
        run_num = args.run
	shot_nums = args.shots.split(' ')
	all_shots = args.all_shots
	data_dir = args.directory
	write_dir = args.write
	est = args.model

        # Call main()
        main(run_num, shot_nums, all_shots, data_dir, write_dir, est)

