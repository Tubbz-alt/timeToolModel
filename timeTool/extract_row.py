import argparse

def write_row_to_col(run, shot, data_dir, write_dir):
	with open(data_dir + '/xppl3816_r' + str(run) + '_matrix.dat','r') as f:
		row = f.readlines()[shot].strip('\n').split(' ')

	with open(write_dir + 'vi/col_data/r' + str(run) + '_s' + str(shot) + '_col.dat','w') as f:
		f.writelines([str(item) + '\n' for item in row])


if __name__=='__main__':

	# Set up arg parser 
        helpstr = 'Write a particular shot signal to column form for gnuplot'
        parser = argparse.ArgumentParser(description=helpstr);
        parser.add_argument('-r','--run', dest='run', type=int, help='run number', default=51)
	parser.add_argument('-s','--shot',dest='shot',type=int, help='shot number', default=0)
	parser.add_argument('-d','--directory',dest='directory', type=str, help='path to directory with original data files', default='/reg/d/psdm/XPP/xppl3816/scratch/timeTool_ml/data_source/')
        parser.add_argument('-w','--write',dest='write',type=str, help='path to directory to write files to', default='/reg/d/psdm/XPP/xppl3816/scratch/timeTool_ml/data_results/')

        args = parser.parse_args();

        # Access input
        run_num = args.run
	shot_num = args.shot
	data_dir = args.directory
	write_dir = args.write

        # Call method
	write_row_to_col(run_num, shot_num, data_dir, write_dir)
	
	
