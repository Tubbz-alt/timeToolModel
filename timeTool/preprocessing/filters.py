import sys
import argparse

# NOTE: running setup.py will ensure full setup of all files required for regression analysis, 
#	whereas this script only creates the filtered files

min = 350		# minimum pixel_pos for optimal training band
max = 352		# maximum pixel_pos for optimal training band
amp_th = 0.05		# threshold for filtering amplitude (all data points with amp > amp_th are used)

#############################################
#Discover shots with pixel pos in [350,352) #
#############################################
def filter_data_by_pixel_pos(run_num, data_dir, write_dir):

	# Open tt.dat file and extract pixel_pos (a.k.a index)
	with open(data_dir + 'xppl3816_r' + str(run_num) + '_tt.dat', 'r') as f:
	        lines = f.readlines()
	pixels = [float(line.split(' ')[0]) for line in lines if not line.startswith('#')]

	# Write filtered shot numbers to file to be used for training with known ground truth	
	with open(write_dir + 'xppl3816_r' + str(run_num) + '_training.dat','w') as f:
		f.writelines(str(i) + '\n' for i,pixel in enumerate(pixels) if min<=pixel<max) 

#############################################
#Access signal,amp,pixel_pos & filter by amp#
#############################################
def filter_data_by_amp(run_num, data_dir, write_dir):

	# Open tt.dat file and filter amplitude and pixels by amplitude
        with open(data_dir + 'xppl3816_r' + str(run_num) + '_tt.dat', 'r') as f:
                amps = [float(line.split(' ')[1]) for line in f.readlines() if not line.startswith('#')]
        
	# Write all filtered data to file 
	with open(write_dir + 'xppl3816_r' + str(run_num) + '_filtered_amp.dat','w') as f:        
		f.writelines(str(i) + '\n' for i,amp in enumerate(amps) if amp > amp_th)

def main(pix_bool, amp_bool, run_num, data_dir, write_dir):

	if pix_bool:	
		filter_data_by_pixel_pos(run_num, data_dir, write_dir)
	
	if amp_bool:

		filter_data_by_amp(run_num, data_dir, write_dir)
	
if __name__ == '__main__': 

	# Set up arg parser 
	helpstr = 'Filter data from a specified run by either pixel_pos or amplitude (or both)--controled by booleans'
        parser = argparse.ArgumentParser(description=helpstr);
        parser.add_argument('-p','--pixel', dest='pixel', type=bool, help='True=filter data by pixel pos', default=False)
        parser.add_argument('-a','--amp', dest='amp', type=bool, help='True=filter data by amplitude', default=True)
        parser.add_argument('-r','--run', dest='run', type=int, help='run number', default=52)
	parser.add_argument('-d','--directory',dest='directory', type=str, help='path to directory with original data files', default='/reg/d/psdm/XPP/xppl3816/scratch/timeTool_ml/data_source/')
        parser.add_argument('-w','--write',dest='write',type=str, help='path to directory to write files to', default='/reg/d/psdm/XPP/xppl3816/scratch/timeTool_ml/data_results/')

	args = parser.parse_args();

	# Access input
        pix_bool = args.pixel
        amp_bool = args.amp
        run_num = args.run
	data_dir = args.directory
	write_dir = args.write

	# Call main()
	main(pix_bool, amp_bool, run_num, data_dir, write_dir)
