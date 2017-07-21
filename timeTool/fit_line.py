from sklearn.linear_model import LinearRegression
import argparse
import math
import numpy as np


#############################################
# Read shot_num,pixel_pos,&predicted delay  #
#############################################
def read_data(run_num):

	# Access predicted data from RF
	with open(write_dir + 'xppl3816_r' + str(run_num) + '_RF_plot.dat','r') as f:
		lines = f.readlines()

	# Build data dictionary to contain shot_num: pixel_pos,delay
	data = {int(float(line.split(' ')[0])):(float(line.split(' ')[2]),float(line.split(' ')[3])) for line in lines } 

	return data

#############################################
#    Split data into their correct steps    #
#############################################
def split_data(data):
	
	step = 2	# start at step=2 since first two steps are skipped due to error in ref image

	# Keep dictionaries of lists of pixel_pos and delay values, whose key is the step number
	partitioned_pixel = {}
	partitioned_delay = {}
	pixel = []
	delay = []
	for key,value in data.iteritems():

		# Detect when a new step is starting via shot num and assign list to dict
		if float(key)/1202.0 > float(step):
			if len(pixel) > 0 and len(delay) > 0:	
				partitioned_pixel[step] = np.array(pixel)
				partitioned_delay[step] = np.array(delay)
				pixel = []
				delay = []
			step += 1

		# Keep track of all data points from the same step 
		pixel.append(value[0])
		delay.append(value[1])

	# Write final list to dictionary
	partitioned_pixel[step] = np.array(pixel)
	partitioned_delay[step] = np.array(delay)

	# Return the partitioned data 
	return [partitioned_pixel, partitioned_delay]

#############################################
#        Fit a line to lists X and Y        #
#############################################
def fit_line(X, Y):

	# Fit line to pixel_pos (x) vs. delay (y) 
	model = LinearRegression()
	model.fit(X.reshape(-1,1),Y.reshape(-1,1))

	# Return the slope and the y intercept 
	return [float(model.coef_[0]), float(model.intercept_[0])]

#############################################
#   Write results of line fitting to file   #
#############################################
# This file can be used to plot the fitted lines on top of the data

def write_to_file(run_num, partitioned_pixels, partitioned_delays, m_list, b_list):
	
	with open(write_dir + 'xppl3816_r' + str(run_num) + '_fitted_lines.dat','w') as f:
		
		# Iterate through every step number 
		for step_num,single_step_pixels in partitioned_pixels.iteritems():
        	        single_step_delays = partitioned_delays[step_num]
			
			# Access slope and y-intercept of fitted line to that step
			m = m_list[step_num-3]
			b = b_list[step_num-3]

			# Write every data point in that step to file
			f.writelines(str(step_num) + ' ' + str(pix_del_pair[0]) + ' ' + str(pix_del_pair[1]) + ' ' + str(m) + ' ' + str(b) + '\n' for pix_del_pair in zip(single_step_pixels,single_step_delays))	
#############################################
#      Write histogram data to files        #
#############################################
def write_histogram_files(run_num, m_list, b_list):

	# 2D histogram: y0_diff vs. m_diff
	m_diff = np.array([math.fabs(m2-m1) for (m1,m2) in zip(m_list[0:-1],m_list[1:])])
	b_diff = np.array([math.fabs(b2-b1) for (b1,b2) in zip(b_list[0:-1],b_list[1:])])

	# Create histogram of b_diff and m_diff
	[count,x_edges,y_edges] = np.histogram2d(b_diff, m_diff)

	# Find midpoints for b_diff and m_diff bins
	x_bins = [(x1+x2)/2.0 for (x1,x2) in zip(x_edges.tolist()[0:-1], y_edges.tolist()[1:])]
	y_bins = [(y1+y2)/2.0 for (y1,y2) in zip(y_edges.tolist()[0:-1],y_edges.tolist()[1:])]
	
	# Write results to file to be plotted
	with open(write_dir + 'xppl3816_r' + str(run_num) + '_histogram_diff.dat','w') as f:
		for i,x in enumerate(x_bins):
			for j,y in enumerate(y_bins):
				f.write(str(x) + ' ' + str(y) + ' ' + str(count[i][j]) + '\n')
			f.write('\n')

	# 1D histogram: m
	[count,edges] = np.histogram(m_list, bins=30)

	# Calculate midpoint for each bin
	bins = [(edge2 + edge1)/2.0 for (edge1,edge2) in zip(edges[0:-1],edges[1:])]

	# Write results to file to be plotted
	with open(write_dir + 'xppl3816_r' + str(run_num) + '_histogram_m.dat','w') as f:
		f.writelines([str(m) + ' ' + str(count[i]) + '\n' for i,m in enumerate(bins)])


def main(run_num, fit_bool, hist_bool, dd, wd):	

	global data_dir
	data_dir = dd
	global write_dir 
	write_dir = wd

	if fit_bool:
		data = read_data(run_num)
		[partitioned_X, partitioned_Y] = split_data(data)

		# Initialize variables designed to calculate mean slope and mean c, as well as tracking every m, b, and c
		m_sum = 0.0	
		m_list = []
		delta_c_sum = 0.0
		b_list = []
		last_c = 1000000

		# Iterate through steps
		for key,value in partitioned_X.iteritems():
			
			# Fit a line to that step's data
			[m,b] = fit_line(value,partitioned_Y[key])
			
			# Calculate f(350)
			c = (m*350.0) + b
			
			# Use for mean m calculation
			m_sum += m
		
			# Use for mean c_diff calculation	
			if last_c < 10000: 
				delta_c_sum += (c - last_c)
			else:
				c_start = c
			last_c = c

			# Keep track of every m and b
			m_list.append(m)
			b_list.append(b)

		# Calculate mean slope and mean delta c
		m = m_sum / float(len(partitioned_X.values()))
		delta_c = delta_c_sum / float(len(partitioned_X.values())-1)
	
		# Print results
		print('Average m: ' + str(m))
		print('Average delta_c: ' + str(delta_c))
		print('Starting c: ' + str(c_start))	
	
		# Write results
		write_to_file(run_num, partitioned_X, partitioned_Y, m_list, b_list)

	# Write histogram files 
	if hist_bool:
		write_histogram_files(run_num, m_list, b_list)

if __name__=='__main__': 

	helpstr = 'Perform analysis on results by fitting lines and drawing histograms'
        parser = argparse.ArgumentParser(description=helpstr);
        parser.add_argument('-r','--run', dest='run', type=int, help='run number', default=52)
	parser.add_argument('-f','--fit', dest='fit', type=bool, help='True=fit line', default=True)
	parser.add_argument('-hs','--hist', dest='hist', type=bool, help='True=draw histogram', default=True)
	parser.add_argument('-d','--directory',dest='directory', type=str, help='path to directory with original data files', default='/reg/d/psdm/XPP/xppl3816/scratch/timeTool_ml/data_source/')
        parser.add_argument('-w','--write',dest='write',type=str, help='path to directory to write files to', default='/reg/d/psdm/XPP/xppl3816/scratch/timeTool_ml/data_results/')

        args = parser.parse_args();

        run_num = args.run
        fit_bool = args.fit
        hist_bool = args.hist
	data_dir = args.directory
	write_dir = args.write

        main(run_num, fit_bool, hist_bool, data_dir, write_dir)
