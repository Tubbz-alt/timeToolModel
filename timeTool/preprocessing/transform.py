import numpy as np
import scipy.signal as sp
from numpy.fft import fft as FFT
from numpy.fft import ifft as IFFT
from numpy.fft import fftfreq as FREQS
from cmath import rect
from math import exp

nprect = np.vectorize(rect)

def read_lineouts(filename):

	return np.loadtxt(filename)

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

	with open(filename, 'w') as f:
		f.writelines([str(s) + '\n' for s in signal])


lineouts = read_lineouts('/reg/d/psdm/XPP/xppl3816/scratch/timeTool_ml/data_source/xppl3816_r51_matrix.dat')

ft_mat = np.zeros((lineouts.shape[0], lineouts.shape[1]*2))

for i,lineout in enumerate(lineouts):

	ft_abs, ft_arg = transform(lineout)

	ft_mat[i,] = np.concatenate((ft_abs, ft_arg))


np.savetxt('/reg/d/psdm/XPP/xppl3816/scratch/timeTool_ml/transformed_source/xppl3816_r51_matrix.dat', ft_mat, delimiter=' ')

