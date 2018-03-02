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

nprect = np.vectorize(rect);

def i2lam(i):
	#lset = 600nm for amox28216 for lots of the runs.  Chedck the spectrometer wavelength
        lset=600; 
        nmPi=0.217;
        seterr=1.0051;
        return nmPi*i + seterr*lset - 110.072;
#'i' is the pixel index [ 0 .. 1023 ] and the wavelength [nm]
# run 31 is 1 micron SiN and shows strong etalon

def signal_weights(w):
	x=np.linspace(0,np.pi,num = w.shape[0],endpoint = False);
	a=500.;
	b=1e3;
	w = np.ones(x.shape,dtype = float);
	w[1:] = np.copy(a*np.power(np.sin(x[1:]),int(-2))+b);
	return w;

def weiner_weights(w):
	x=np.linspace(0,np.pi,num = w.shape[0],endpoint = False);
	a=500.;
	b=1e3;
	signalPnoise = np.ones(x.shape,dtype=float);
	signalPnoise[1:]=a*np.power(np.sin(x[1:]),int(-2))+b;
	w = np.copy((signalPnoise-np.min(signalPnoise))/signalPnoise);
	return w;
	#sigPnoise(x)=5e2*(1/s(x))+1e3
	#sigPnoise(x)=5e2*(1/s(x))+1e3

def fourier_reshape(vec):
	vecFT = np.array(vec.shape,dtype=complex);
	dvecFT = np.array(vec.shape,dtype=complex);
	ddvecFT = np.array(vec.shape,dtype=complex);
	dddvecFT = np.array(vec.shape,dtype=complex);
	vecFT = FFT(vec);
	freqs = FREQS(len(vec))	
	dvecFT = 1j*(freqs+.01j)*vecFT;
	ddvecFT = -1*np.power(freqs+.01j,int(2))*vecFT;
	dddvecFT = -1j*np.power(freqs+.01j,int(3))*vecFT;
	dvec = np.real(IFFT(dvecFT))/vec.shape[0];	
	ddvec = np.real(IFFT(ddvecFT))/vec.shape[0];	
	dddvec = np.real(IFFT(dddvecFT))/vec.shape[0];	
	return dvec,ddvec,dddvec;

def rewrap(vec,low,high,step):
	inds = np.argwhere(vec>=high);	
	while len(inds)>0:
		vec[inds] -= step;
		inds = np.argwhere(vec>=high);	
	inds = np.argwhere(vec<low);	
	while len(inds)>0:
		vec[inds] += step;
		inds = np.argwhere(vec<low);	
	return vec;

def between(val,low,high):
	return np.logical_and(val>=low,val<high);

def chooseslope(v,n):
	vec = np.copy(v);
	step = 2.*np.pi/n;
	wholestep = 2*np.pi;
	win = 0.05*step;
	"""
	Need to start counting imposed offsets.
	"""
	if between(vec[0],-step,0):
		if between(vec[1]+step,vec[0]-win,vec[0]+win):
			vec[1] += step;
			if between(vec[2]+2*step,vec[1]-win,vec[1]+win):
				vec[2] += 2*step;
				if between(vec[3]+3*step - wholestep,vec[2]-win,vec[2]+win):
					vec[3] += 3*step - wholestep;
					if between(vec[4]+4*step - wholestep,vec[3]-win,vec[3]+win):
						vec[4] += 4*step - wholestep;
						return np.mean(vec[:5]);
					return np.mean(vec[:4]);
				return np.mean(vec[:3]);
			return np.mean(vec[:2]);
		return np.mean(vec[:1]);
	else:
		vec = np.roll(vec,1);
		if between(vec[0] - step,-2*step,-1*step):
			vec[0] -= step;
			if between(vec[1] + 0*step - wholestep, vec[0]-win,vec[0]+win):
				vec[1] += 0*step - wholestep;
				if between(vec[2] + 1*step - wholestep, vec[1]-win,vec[1]+win):
					vec[2] += 1*step - wholestep;
					if between(vec[3] + 2*step - wholestep, vec[2]-win,vec[2]+win):
						return np.mean(vec[:4]);
					return np.mean(vec[:3]);
				return np.mean(vec[:2]);
			return np.mean(vec[:1]);
		else:
			vec = np.roll(vec,1);
			if between(vec[0] + -2*step, -3*step, -2*step):
				vec[0] -= 2*step ;
				if between(vec[1] + 0*step - wholestep, vec[0]-win,vec[0]+win):
					vec[1] -= wholestep;
					if between(vec[2] + 1*step - wholestep, vec[1]-win,vec[1]+win):
						vec[2] += 1*step - wholestep;
						if between(vec[3] + 2*step - wholestep, vec[2]-win,vec[2]+win):
							vec[3] += 2*step - wholestep;
							if between(vec[4] + 3*step - wholestep, vec[3]-win,vec[3]+win):
								vec[4] += 3*step - wholestep;
								return np.mean(vec[:5]);
							return np.mean(vec[:4]);
						return np.mean(vec[:3]);
					return np.mean(vec[:2]);
				return np.mean(vec[:1]);
			else:
				vec = np.roll(vec,1);
				if between(vec[0] -3*step , -4*step, -3*step):
					vec[0] -= 3*step;
					if between(vec[1] -2*step , vec[0]-win,vec[0]+win):
						vec[1] -= 2*step;
						"""
						I think I'm lost here.
						"""
						if between(vec[1] -2*step , vec[0]-win,vec[0]+win):
							vec[2] -= wholestep;
							return np.mean(vec[:3]);
						return np.mean(vec[:2]);
					return np.mean(vec[:1]);
				else:
					return 20.
	return 10.;

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


