import numpy as np;
from numpy.fft import fft as FFT;
from numpy.fft import ifft as IFFT;
from numpy.fft import fftfreq as FREQS;

# Determinstic algorithm!!!
# -------------------------
# Inputs: a lineout that should be an array of length n containing a signal from a detector image.
# 			You can optionally provide the number of rolls, but 25 seems pretty good for now.
# Outputs: a magic number that correlates to the true delay for the shot, computed using FFT and rolling.
#
# This can be combined with the ground truth generator to provide a corrected detector offset.


# This is the only function you should call!
def compute(lineouts, nrolls = 25):

	slopes = np.zeros(lineouts.shape[0])
	for i,lineout in enumerate(lineouts):
		slopes[i] = __compute(lineout, nrolls)

	return slopes

def __compute(lineout, nrolls = 25):

	rolls = __generateRolls(lineout, nrolls)

	rollerFT = FFT(rolls,axis=1);
	ft_abs = np.copy(np.abs(rollerFT[1,:]));
	dargs = np.diff( np.unwrap( np.angle( rollerFT ) , axis = 1 ), axis =1 );
	avg = np.average(dargs,axis=1,weights = ft_abs[1:]);
	toReturn = __chooseRoll(avg, nrolls);

	return toReturn

# see the underscores?? those mean don't call me on my own!
def __generateRolls(lineout, nrolls):
	roller = np.zeros((nrolls,len(lineout)),dtype=float);
	for r in range(nrolls):
		roller[r,:] = np.roll(np.copy(lineout), r*len(lineout)//nrolls);

	return roller

# like above, underscores means don't touch
def __chooseRoll(avg, nrolls):
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

	if bestGuess > 0:
		bestGuess -= 2.*np.pi;

	return bestGuess;

