import numpy as np
import randomForestRegressor
import linearRegressor
import supportVectorRegressor
import argparse
from sklearn.metrics import r2_score
from tqdm import tqdm
import datetime

shot_num_th = 2404

############################################
#      Access all data filtered by amp     #
############################################
def access_data(data_dir, write_dir, run_num, shot_th):
	
	if shot_th > 0:
		# Access shot numbers > shot_th filtered by amplitude
	        with open(write_dir + 'xppl3816_r' + str(run_num) + '_filtered_amp.dat', 'r') as f:
        	        lines = f.readlines()

		filtered_shots = [int(float(line.strip('\n'))) for line in lines if int(float(line.strip('\n'))) > shot_th]
	else:

		# Access shot numbers < shot_th with pixel_pos in [350,352)
		with open(write_dir + 'xppl3816_r' + str(run_num) + '_training.dat','r') as f:
			small_shots_train = set([int(float(line.strip('\n'))) for line in f.readlines() if int(float(line.strip('\n'))) <= shot_num_th])
	
		# Access shot numbers > shot_th filtered by amplitude
        	with open(write_dir + 'xppl3816_r' + str(run_num) + '_filtered_amp.dat', 'r') as f:
                	lines = f.readlines()		
		
		small_shots_test = set([int(float(line.strip('\n'))) for line in lines if int(float(line.strip('\n'))) <= shot_num_th])
		large_shots = [int(float(line.strip('\n'))) for line in lines if int(float(line.strip('\n'))) > shot_num_th]

		# Find shot numbers < shot_th with pixel_pos in [350,352) AND with sufficient amplitudes, then union with large shot numbers
		filtered_shots = list(small_shots_train & small_shots_test) + large_shots

	# Access data matrix, amplitudes and pixels 
	with open(data_dir + 'xppl3816_r' + str(run_num) + '_matrix.dat','r') as f:
		unfiltered_input = [line for line in f.readlines() if not line.startswith('#')]
	with open(data_dir + 'xppl3816_r' + str(run_num) + '_tt.dat','r') as f:
		unfiltered_tt = [(float(line.split(' ')[1]), float(line.split(' ')[0])) for line in f.readlines() if not line.startswith('#')]

	# Filtered based on filtered_shots
	filtered_tt = {}
	filtered_data = []
	for shot in filtered_shots:
		filtered_tt[shot] = unfiltered_tt[shot]
		filtered_data.append([float(x) for x in unfiltered_input[shot].strip('\n').split(' ')])
	filtered_inputs = np.matrix(filtered_data)

        return [filtered_inputs, filtered_tt]

############################################
#   Read ground truth delays of test       #
############################################
def read_ground_truth(write_dir, run_num, shot_th):

	if shot_th > 0:
		with open(write_dir + 'xppl3816_r' + str(run_num) + '_RF_plot.dat','r') as f:
	                test_truths = np.array([float(line.strip('\n').split(' ')[3]) for line in f.readlines() if float(line.strip('\n').split(' ')[0]) > shot_th])

	else:

		# Read prediction of RF when trained on pixel_pos band and tested on entire dataset for that run
		with open(write_dir + 'xppl3816_r' + str(run_num) + '_RF_plot.dat','r') as f:
			test_truths = np.array([float(line.strip('\n').split(' ')[3]) for line in f.readlines()])	

	return test_truths

############################################
#          Apply model to data             #
############################################
def apply_model(estimator, data):
	
	y_pred = estimator.predict(data)

	return y_pred

############################################
# Plot shot_num x pixel_pos x corrected_y  #
############################################
def write_plot_file(write_dir, filtered_tt, y_pred, run_num, est, ci):
	with open(write_dir + 'xppl3816_r' + str(run_num) + '_' + est + '_plot.dat', 'w') as f:
		f.writelines([str(shot) + ' ' + str(amp) + ' ' + str(pixel) + ' ' + str(y_pred[i]) + ' ' + str(y_pred[i] - ci[i][0]) + ' ' + str(ci[i][1] - y_pred[i]) + '\n' for i,(shot,(amp,pixel)) in enumerate(filtered_tt.iteritems())])		

############################################
#    Write comp. time + R2 value           #
############################################
def write_performance_file(write_dir, run_num, train_time, test_time, r2, rand_int):
	with open(write_dir + 'xppl3816_r' + str(run_num) + '_performance.dat', 'a') as f:
		f.write(str(rand_int) + ' ' + str(train_time) + ' ' + str(test_time) + ' ' + str(r2) + '\n')

def main(train_num, test_num, est, data_dir, write_dir, rand_int):

	# If training & testint on the same run
	if test_num is None: 
		single_run = True
		test_num = train_num
	# Otherwise it is a run to run test
	else:
		single_run = False		

	# Access regressor that was selected
	if est == 'RF':
		estimator_type = randomForestRegressor
	elif est == 'LR':
		estimator_type = linearRegressor
	elif est == 'SVM':
		estimator_type = supportVectorRegressor

	# Read training data (filtered by amplitude)
	print('Reading training data')
	if single_run:
		[training_data, training_truths, n, m] = estimator_type.read_training_data(data_dir, write_dir, train_num)
		print(m)
	else:
		[training_data, x] = access_data(data_dir,write_dir,train_num, 0)
		training_truths = read_ground_truth(write_dir, train_num, 0)

	# Train model
	print('Training ' + est)
	train_start = datetime.datetime.now()
	estimator = estimator_type.train(training_data, training_truths, rand_int)
	train_time = (datetime.datetime.now() - train_start).total_seconds()

	# Filter test data (filtered by amplitude)
	print('Reading test data')
	if single_run:
		[filtered_inputs, filtered_tt] = access_data(data_dir,write_dir,test_num, 0)
	else:
		[filtered_inputs, filtered_tt] = access_data(data_dir,write_dir,test_num, shot_num_th)

	# Apply model to test data
	print('Appying ' + est + ' to filtered test data')
	test_start = datetime.datetime.now()
	y_pred = apply_model(estimator, filtered_inputs)
	test_time = (datetime.datetime.now() - test_start).total_seconds()

	# Calculcate confidence interval
	ci = randomForestRegressor.calculcate_ci(estimator, filtered_inputs)

	# Write results to file
	if single_run:
		run_num = str(train_num)
	else:
		run_num = str(train_num) + '_' + str(test_num)
		test_truths = read_ground_truth(write_dir, test_num, shot_num_th)
		r2 = r2_score(test_truths,y_pred)
		print('R2 when trained on ' + str(train_num) + ' and tested on ' + str(test_num) + ' :' + str(r2)) 

		write_performance_file(write_dir, run_num, train_time, test_time, r2, rand_int)
	
	write_plot_file(write_dir, filtered_tt, y_pred, run_num, est, ci)

if __name__ == "__main__":
	
	helpstr = 'Train and test regression model on one or more runs'
        parser = argparse.ArgumentParser(description=helpstr);
        parser.add_argument('-r','--train', dest='train', type=int, help='training data run number', default=51)
        parser.add_argument('-t','--test', dest='test', type=int, help='testing data run number', default=None)
        parser.add_argument('-m','--model', dest='model', type=str, help='type of regression model (RF for random forest, LR for linear regression, SVM for support vector machine)', default='RF')
	parser.add_argument('-d','--directory',dest='directory', type=str, help='path to directory with original data files', default='/reg/d/psdm/XPP/xppl3816/scratch/timeTool_ml/data_source/')
        parser.add_argument('-w','--write',dest='write',type=str, help='path to directory to write files to', default='/reg/d/psdm/XPP/xppl3816/scratch/timeTool_ml/data_results/')
 	parser.add_argument('-i','--randint',dest='randint',type=int, help='seed for random forest',default=2)
	args = parser.parse_args();

        train_num = args.train
        test_num = args.test
	est = args.model
	data_dir = args.directory
	write_dir = args.write
	randint = args.randint

	main(train_num, test_num, est, data_dir, write_dir, randint)
