import math
import randomForestRegressor 
import linearRegressor
import argparse
from random import randint
import numpy as np
import extract_row as er
import os

# Read all shots from a run
def access_data(data_dir, run_num):
	with open(data_dir + 'xppl3816_r' + str(run_num) + '_matrix.dat', 'r') as f:
                lines = f.readlines()
        signals = np.matrix([[float(x) for x in line.strip('\n').split(' ')] for line in lines if not line.startswith('#')])

	return signals

# Calculate variable importance for every feature
def calculate_vi(model, signals, shot_num):

        # Access original signal and perform regression on it 
        signal = signals[shot_num]
        original_pred = model.predict(signal.reshape(1, -1))

        # Track max difference in prediction and the corresponding index
        pred_diff = {}

        # For each feature, replace it with the feature from a randomly selected data point 3 times
        for i in range(0, signal.shape[1]):

		pred_diff_sum = 0.0

		# Permute 3 times for each feature and average for more precise answer
		for j in range(0,3):
                
			# Access original signal then replace feature value with randomly selected shot
			permuted_signal = signal
                	permuted_signal.put(i, signals[randint(0,signals.shape[0]-1)].item(i))

			# Calculate difference in prediction with permutation
                	new_pred = model.predict(permuted_signal.reshape(1, -1))
                	pred_diff_sum += math.fabs(original_pred - new_pred)
		
		# Average 3 instances
		pred_diff[i] = pred_diff_sum / 3.0

        return list(pred_diff.values())

# Write file containing variable importance of each feature to file
def write_vi(write_dir, run_num, shot_num, pred_diff):

	with open(write_dir + 'vi/vi_data/r' + str(run_num) + '_s' + str(shot_num) + '_vi.dat','w') as f:
		f.writelines([str(diff) + '\n' for diff in pred_diff])

# Write gnuplot file to display results overlaid on signal
def write_gnuplot(run_num, shot_num, write_dir):

	with open(write_dir + 'vi/vi.gp','w') as f:

		f.write('set term png\n')
		f.write('set output \'../work/transferLearning/pngs_to_label/r' + str(run_num) + '_s' + str(shot_num) + '_vi.png\'\n')
		f.write('set xlabel \'index\'\n')
		f.write('set ylabel \'signal\'\n')
		f.write('set cblabel \'prediction error\'\n')
		f.write('set title \'Variable importance results for run ' + str(run_num) + ', shot ' + str(shot_num) + '\'\n')
		f.write('plot \"< paste \'' + write_dir + 'vi/col_data/r' + str(run_num) + '_s' + str(shot_num) + '_col.dat\' \'' + write_dir + 'vi/vi_data/r' + str(run_num) + '_s' + str(shot_num) + '_vi.dat\'\" u 0:1:2 palette lw 2 with lines title \'opal_0 signal\'\n')
	
	# Run gnuplot script to generate png
	os.system('gnuplot ' + write_dir + 'vi/vi.gp')

def main(run_num, shot_num, data_dir, write_dir, inc, est):

	# Access regressor that was selected
        if est == 'RF':
                estimator_type = randomForestRegressor
        elif est == 'LR':
                estimator_type = linearRegressor
	
	# Access training data -- that is, data in pixel_pos range [350,352)
        [inputs, ground_truths, num_samples, num_features] = estimator_type.read_training_data(data_dir, write_dir, run_num)
	
	# Train forest on training data
	forest = estimator_type.train(inputs, ground_truths)

	# Access all data 
	signals = access_data(data_dir, run_num)

	# Inc is increment of shots (i.e. every 100 shots)
	if inc is None:
		shot_nums = [shot_num]
	else:
		shot_nums = [i for i in range(0,signals.shape[0],inc)]

	# Loop through all specified shot numbers
	for shot_num in shot_nums:
		# Calculate variable importance for all features
		importances = calculate_vi(forest, signals, shot_num)

		# Write vi results to file
		write_vi(write_dir, run_num, shot_num, importances)

		# Write shot row data to column format for gnuplot
                er.write_row_to_col(run_num, shot_num, data_dir, write_dir)

		# Write gnuplot file
		write_gnuplot(run_num, shot_num, write_dir)		

if __name__=="__main__": 

	# Set up arg parser 
        helpstr = 'Determine most import variable in a particular data point'
        parser = argparse.ArgumentParser(description=helpstr);
        parser.add_argument('-r','--run', dest='run', type=int, help='run number', default=52)
	parser.add_argument('-s','--shot',dest='shot', type=int, help='shot number', default=0)
	parser.add_argument('-i','--inc', dest='increment', type=int, help='every __ shots', default=None)
	parser.add_argument('-d','--directory',dest='directory', type=str, help='path to directory with original data files', default='/reg/d/psdm/XPP/xppl3816/scratch/timeTool_ml/data_source/')
        parser.add_argument('-w','--write',dest='write',type=str, help='path to scratch directory where files are  written to', default='/reg/d/psdm/XPP/xppl3816/scratch/timeTool_ml/data_results/')
	parser.add_argument('-m','--model',dest='model',type=str, help='regression model (RF or LR)', default='RF')

        args = parser.parse_args();

        # Access input
        run_num = args.run
	shot_num = args.shot
	data_dir = args.directory
	write_dir = args.write
	inc = args.increment
	est = args.model

        # Call main()
        main(run_num, shot_num, data_dir, write_dir, inc, est)

