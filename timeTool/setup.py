import argparse
import filters
import regression

# This script sets up all files necessary to run regression on either one run or on multiple

def main(run_nums, data_dir, write_dir):

	print('Setting up files for runs ' + str(run_nums))

	for run_num in run_nums.split(' '):

		print('Run: ' + str(run_num))
	
		# Set up filtered data files
		print('Creating filtered files...')
		filters.main(True, True, run_num, data_dir, write_dir)	

		# Run RF regression on each run to obtain ground truth files
		print('Creating ground truth files...')
		regression.main(run_num, None, 'RF', data_dir, write_dir, 2)

		print('\n')

	print('Done')

if __name__=='__main__': 

	helpstr = 'Setup all files needed for regression on either one run or multiple'
        parser = argparse.ArgumentParser(description=helpstr);
        parser.add_argument('-r','--run', dest='runs', type=str, help='data run number(s) to set up', default="17 51 52 106 110")
	parser.add_argument('-d','--directory',dest='directory', type=str, help='path to directory with original data files', default='/reg/d/psdm/XPP/xppl3816/scratch/timeTool_ml/data_source/')
        parser.add_argument('-w','--write',dest='write',type=str, help='path to directory to write files to', default='/reg/d/psdm/XPP/xppl3816/scratch/timeTool_ml/data_results/')       
 
	args = parser.parse_args();

        run_nums = args.runs
	data_dir = args.directory
	write_dir = args.write

	main(run_nums, data_dir, write_dir)
