import numpy as np
import scipy.signal as sp
from numpy.fft import fft as FFT
from numpy.fft import ifft as IFFT
from numpy.fft import fftfreq as FREQS
from cmath import rect
from math import exp
from randomForestRegressorSim import read_training_data 
from andiff_filters_sim import write_all_files

nprect = np.vectorize(rect)

def read_lineouts(data_dir):

	lineouts, delays, n_sample, n_features, shots = read_training_data(data_dir)

	return lineouts, n_features

def transform(lineout):

	lineoutFT = FFT(lineout)

	return np.abs(lineoutFT), np.unwrap(np.angle(lineoutFT))

def my_weiner_filter(signal):

	n = 1e2
	a = 1e6
	w = 50
	e = 0.25

	freq = FREQS(len(signal))
	f = a*(-(freq**2/w)**e)

	filtered_signal = signal**2 * (f / np.add(f, n))

	return filtered_signal, f/np.add(f,n)

def sp_weiner_filter(signal, mysize=50, noise=1e2):

	return sp.wiener(signal)

def back_transform(ft_abs, ft_arg):

	return np.real(IFFT(nprect(ft_abs,ft_arg))) 

def write_signal(signal, filename):

	with open(filename, 'w') as f:
		f.writelines([str(s) + '\n' for s in signal])

directory_in = 'data_simulation/'
directory_out = '/reg/d/psdm/XPP/xppl3816/scratch/data_simulation_filtered/wiener/'

lineouts, n_features = read_lineouts(directory_in)

filtered = np.zeros((0,n_features),dtype=float)

for lineout in lineouts:

	###########################
	# Scipy's Wiener filter   #
	###########################

	filtered = np.row_stack((filtered,sp_weiner_filter(lineout)))

write_all_files(filtered, directory_in, directory_out, 0)
		
