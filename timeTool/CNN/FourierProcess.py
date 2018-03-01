import numpy as np;
from numpy.fft import fft as FFT;
from numpy.fft import ifft as IFFT;
from numpy.fft import fftfreq as FREQS;
import math;

class FourierProcess:

	lines = [(271, 278), (351, 358), (362, 369), (407, 415), (441, 449), (476, 483), (544, 551), (681, 688), (750, 757)];
	noiseLevel = 1e-3
	startCos2 = 30
	endCos2 = 60
	nbins = 200

	def __init__(self, inputImage):
		self.image = inputImage

	def process(self):
		detectedVals = np.zeros(len(self.lines));

		for lidx, l, in enumerate(self.lines):

			line = self.image[l[0]:l[1] + 1,:];
			meanLine = np.mean(line, axis=0);

			FOURIER_LINE = FFT(meanLine)
			FILTERED = self.weinerFilter(FOURIER_LINE);
			PHASE = self.extractPhaseSlope(FILTERED);
		
			detectedVals[lidx] = self.extractMaxPhase(PHASE);

		return detectedVals;

	def weinerFilter(self, SIGNAL):

		WEINER_S = np.zeros(len(SIGNAL))
		WEINER_S[1:self.startCos2] = 1
		WEINER_S[self.startCos2:self.endCos2] = [pow(math.cos(x), 2.0) for x in np.linspace(0,math.pi / 2.0,self.endCos2 - self.startCos2)]

		WEINER_N = np.ones(len(SIGNAL)) * self.noiseLevel

		WEINER_FILTER = WEINER_S / (WEINER_S + WEINER_N)
		FILTERED_SIGNAL = np.multiply(SIGNAL, WEINER_FILTER)
		
		return FILTERED_SIGNAL

	def extractPhaseSlope(self, SIGNAL):
		PHASE = np.unwrap(np.angle(SIGNAL))
		PHASE_DIFF = np.diff(PHASE)
		PWR = np.power(np.abs(SIGNAL), 2.0)[:-1]
		WEIGHTED_PHASE, bins = np.histogram(PHASE_DIFF,bins=self.nbins,range=(-np.pi,np.pi),weights=PWR, density=True);

		return WEIGHTED_PHASE

	def extractMaxPhase(self, PHASE_SIGNAL):
		phaseBins = np.arange(self.nbins).astype(float);
		estimatedVal = float(np.argmax(PHASE_SIGNAL)) * (float(self.image.shape[1]) / float(self.nbins));

		return estimatedVal

img = np.loadtxt('/reg/neh/home/timaiken/data_2018/amox28216_r74_image200.dat')
p = FourierProcess(img)
vals = p.process()
print(vals)
