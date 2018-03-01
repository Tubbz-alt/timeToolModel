import numpy as np
import scipy.signal as sp
from numpy.fft import fft as FFT
from numpy.fft import ifft as IFFT
from numpy.fft import fftfreq as FREQS
from cmath import rect
from math import exp

nprect = np.vectorize(rect)

def read_lineouts(filename):

	return np.loadtxt(filename, skiprows=30000)

def transform(lineout):

	lineoutFT = FFT(lineout)

	return np.abs(lineoutFT), np.unwrap(np.angle(lineoutFT))

def gram_schmidt(signal, order):

	freq = FREQS(len(signal))	

	diffs = np.zeros((order+1, len(signal)))
	diffs[0,] = signal / np.linalg.norm(signal) 				#a
	
	for i in range(1,order+1):
		diffs[i,] = np.multiply(diffs[i-1,], freq)		#b = a*f, c = b*f
		for j in range(order):
			diffs[i,] -= np.multiply(np.dot(diffs[i,], diffs[j,]), diffs[j,]) # - (cdota)*a - (cdotb)*b  

		diffs[i,] = diffs[i,] / np.linalg.norm(diffs[i,])

	return diffs

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

lineouts = read_lineouts('/reg/d/psdm/XPP/xppl3816/scratch/timeTool_ml/data_source/xppl3816_r110_matrix.dat')

#for lineout in lineouts:
for lineout in lineouts[0:1]:

	write_signal(lineout, 'sig.txt')

	ft_abs, ft_arg = transform(lineout)
	write_signal(ft_abs, 'abs.txt')
	write_signal(ft_arg, 'arg.txt')

	diffs = gram_schmidt(ft_abs, 2)

	weights = np.zeros(diffs.shape[0]*diffs.shape[1])
	for i in range(diffs.shape[0]):
		weights[diffs.shape[1]*i:diffs.shape[1]*(i+1),] = diffs[i,]
		
	write_signal(weights, 'init_weights.txt')

	###########################
	#     My Wiener filter    #
	###########################

	#wdiff,filt = my_weiner_filter(diffs[1,])

	#write_signal(np.concatenate((-wdiff[512:],wdiff[:512])), 'wdiff.txt')
	#write_signal(np.array(range(-511,512)), 'indices.txt')
        #write_signal(np.concatenate((-diffs[1,512:],diffs[1,:512])), 'diff.txt')
	#write_signal(np.concatenate((filt[512:],filt[:512])), 'filter.txt')

        #lineoutback = back_transform(wdiff, ft_arg)
        #write_signal(lineoutback, 'back.txt')


	###########################
	# Scipy's Wiener filter   #
	###########################

	diffback = back_transform(diffs[1,], ft_arg)
	wdiffback = sp_weiner_filter(diffback)
	wdiffabs, wdiffarg = transform(wdiffback)

	write_signal(np.array(range(-511,512)), 'indices.txt')
	write_signal(np.concatenate((-diffs[1,512:],diffs[1,:512])), 'diff.txt')
	write_signal(np.concatenate((-wdiffabs[512:],wdiffabs[:512])), 'wdiff.txt')

	write_signal(wdiffback, 'back.txt')
	
