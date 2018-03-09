#!/reg/g/psdm/sw/releases/ana-current/arch/x86_64-rhel7-gcc48-opt/bin/python

from randomForestRegressorSim import read_training_data
import numpy as np;
from numpy.fft import fft as FFT;
from numpy.fft import ifft as IFFT;
from numpy.fft import fftfreq as FREQS;
import matplotlib.pyplot as plt;
from cmath import rect;
import math
from sklearn.metrics import mean_squared_error as mse

def timsChoice(avg, nrolls):
	step = 2 * np.pi / nrolls;
	delta = 0.1 * step;

	avgNorm = np.array([val + i * step - 2 * np.pi if i * step > np.pi else val + i * step for i, val in enumerate(avg)]);

	numInBestCluster = 0;
	bestGuess = 20;

	for val in avgNorm:
		similarVals = avgNorm[np.where(abs(avgNorm - val) < delta)];
		if len(similarVals) > numInBestCluster:
			numInBestCluster = len(similarVals);
			bestGuess = np.average(similarVals);

	#print(bestGuess)

	if bestGuess > 0:
		bestGuess -= 2.*np.pi;

	return bestGuess;

def lineoutToLocation(lineout, nrolls, nsamples):
	roller = np.zeros((nrolls,nsamples),dtype=float);
	for r in range(nrolls):
		roller[r,:] = np.roll(np.copy(lineout), r*len(lineout)//nrolls);
	rollerFT = FFT(roller,axis=1);
	ft_abs = np.copy(np.abs(rollerFT[1,:]));
	dargs = np.diff( np.unwrap( np.angle( rollerFT ) , axis = 1 ), axis =1 );
	avg = np.average(dargs,axis=1,weights = ft_abs[1:]);
	fourierLocation = timsChoice(avg, nrolls);
	return fourierLocation;
	#pixelGuess = -fourierLocation * len(lineout) / (2 * np.pi)

	#return pixelGuess

	#polyToSolve = np.array(self.params.polyFitParams);
	#polyToSolve[3] -= fourierLocation;
	#roots = np.roots(polyToSolve);

	#for r in roots:
	#        if r > -4 and r < 4 and not np.iscomplex(r):
	#                return r;

	#return None;

data_dir = 'data_simulation/'

lineouts, ground_truths, num_observations, nsamples, shots = read_training_data(data_dir)

np.savetxt(data_dir + 'delays/ground_truth.dat', ground_truths, fmt='%.6e')

num = 0.0;
nrolls = 20;

D = np.zeros((0,1),dtype=float)

for lineout in lineouts:
	d_data = lineoutToLocation(lineout, nrolls, nsamples)
	D = np.row_stack((D,d_data))

print 'Tim\'s RMSE' 
print math.sqrt(mse(ground_truths, D))

dirstr = 'data_simulation/delays/'
filename_tim=dirstr + 'tims_delays.dat'
np.savetxt(filename_tim,D,fmt='%.6e')
	
print('Done.');


