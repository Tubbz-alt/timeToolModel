import numpy as np
import os

A = np.load('../work/SSLearningPipeLine/indexlist.npy')

i = 0
for idx in A:
	if i < 36:
		pass
	else:	
		print('Calculating variable importance for run 110, shot ' + str(idx))
		try: 
			os.system('python variable_importance.py -r 110 -s ' + str(idx))
		except:
			break
	i += 1
