import numpy as np
from itertools import islice
from scipy import signal
import argparse
from os import listdir
from os.path import isfile, join

import sys
sys.path.insert(0, '/reg/neh/home/kfotion/timeTool/RF')

from randomForestRegressorSim import read_all_files

#Last modified 1/12/18



#This program pulls in data sets from experimental runs at 1000 chunks at a time and performs several operations. The first operation, is the data is Fourier transformed, then using a Fourier Spectral Method the derivative is taken of the data chunk across each row of the numpy matrix. Two different analog filters are then applied to the data set, the first is a high cutoff frequency filter, the second is a analog filter with a low cutoff frequency. The Inverse Fourier Transform is taken and the product of these two filters is performed to get the desired output of clean peaks corresponding to the edges of the shot data. HIGHLY RECOMMENDED that the user of this program (whoever you may be :D) become familiar with Fourier Spectral Methods as the program relies on them completeley, THIS IS NOT A OVERSTATEMENT!!!!!



#Down samples array by picking sample size based off cutoff frequency.
def down_sample(arr,fl):
    spacing = int(round((arr.shape[0]*.2)/(fl)))
    return arr[0::spacing]

#Some of the experimental runs have a header line begining with a #. This function creates a boolean that returns true if such a line is found.
def check_header(filen):
    with open(filen, 'r') as f:
        line = f.readline()
        if line[0] == '#':
            return True

#This fuction takes the derivative of the array based on a Fourier Spectral Method.
def der(arr,freqs):
    arr1 = np.multiply(arr,freqs)
    for k in range(len(arr)):
        arr1[k] *= 1j   
    return arr1

#This function creates a simulated analog filter based of a sinc function and a Blackman window. If you want to know more on analog filters there is a great resources online on their theory. The function takes in b = transition band, fl = cutoff frequency, fs = sampling rate.  
def analog(b, fl, fs):
    b /= fs
    N =int(np.ceil(4/b))
    if(N % 2 == 0):
        N += 1
    n = np.arange(N)
    h = np.sinc(2 *fl/fs*(n -(N-1)/2.))
    h *= np.blackman(N)
    h /= np.sum(h) 
    return h

def filter(arr, directory_in, directory_out, row):

    #This loop sets up the file and filter parameters. The file directories are listed as fileo and are found in the directories listed at the time of writing for this code. The main body of this code runs 7 times. More filters can easily be added, all you have to do is add them in the if: statement (At this point should be a switch statement XD) and change the range for the for loop.

    cutoffs = [6, 14, 35, 52, 67, 108, 195]

    for x, fl1 in enumerate(cutoffs): 

        if x == 0:
            b1 = float(50)
            fl2 = float(14)
            b2 = float(13)
        elif x == 1:
            b1 = float(29)
            fl2 = float(8)
            b2 = float(8)
        elif x == 2: 
            b1 = float(5)
            fl2 = float(2.5)
            b2 = float(3)
        elif x == 3:
            b1 = float(13)
            fl2 = float(2.5)
            b2 = float(3)
        elif x == 4:
            b1 = float(180)
            fl2 = float(14)
            b2 = float(13)
        elif x == 5:
            b1 = float(91)
            fl2 = float(14)
            b2 = float(13)
        elif x == 6: 
            b1 = float(42)
            fl2 = float(6)
            b2 = float(5)

        #Create the analog filters based on the parameters above.
        analog14 = analog(b1,fl1,450.0)
        analog195 = analog(b2,fl2,450.0)

        fft = np.fft.rfft(arr, None, 1) #Take real Fourier transform.

        freqs = np.fft.fftfreq(arr.shape[1]+1,1.0) #Create frequency spectrum for derivative
        
        split = np.split(freqs,2)[0] #Split the spectrum in half as the real Fourier transform was taken (There is a function in newer versions of numpy called fft.rfftfreqs that will do this).
        der1 = np.apply_along_axis(der,1,fft,split) #Apply the derivative.
        derfull = np.fft.irfft(der1, None, 1) #Take the inverse Fourier transform.
        datan1 = np.apply_along_axis(lambda m: signal.fftconvolve(m,analog14, mode = "same"), axis =1, arr = derfull) #Apply the analog filters (ironically using fftconvolve).
        datan2 = np.apply_along_axis(lambda m: signal.fftconvolve(m,analog195, mode = "same"), axis =1, arr = derfull)
        final = np.multiply(datan2,np.abs(datan1)) #Multiply the two filters together.
        sampled_data = np.apply_along_axis(down_sample, 1, final,fl1) #Down sample the product of the two filters.
    
        write_all_files(sampled_data, directory_in, directory_out + 'cutoff_' + str(fl1) + '/', row)
    
def write_all_files(data, directory_in, directory_out, row):

    filenames = [f for f in listdir(directory_in) if isfile(join(directory_in, f)) and not f.startswith('.')]

    print(data.shape)
    print(len(filenames))

    for i,filename in enumerate(filenames):

        with open(directory_in + filename, 'r') as f:
            lines = f.readlines()

        line = ''
        for j,x in enumerate(data[i,]):
            if j != 0:
                line += '\t'
            line += str(x)
        line += '\n'

        lines[row+7] = line

        with open(directory_out + filename, 'w') as f:
            f.writelines(lines[:row+8])

def main(args):

    signals, delays, shots = read_all_files(args.directory, args.row)	

    inputs = np.matrix(signals)

    filter(inputs[:,:-1], args.directory, args.out, args.row)

if __name__ == "__main__":

    # Set up arg parser 
    helpstr = 'Filter differentiated signals with various cutoff frequencies'
    parser = argparse.ArgumentParser(description=helpstr);
    parser.add_argument('-d','--directory',dest='directory', type=str, help='path to directory with original data files', default='../data_simulation/')
    parser.add_argument('-o', '--out', dest='out', type=str, help='name of output directory', default='/reg/d/psdm/XPP/xppl3816/scratch/data_simulation_filtered/')
    parser.add_argument('-r', '--row', dest='row', type=int, help='row number from simulation data', default=0)

    args = parser.parse_args();

    # Call main()
    main(args)

