import numpy as np
import matplotlib.pyplot as plt


class GroundTruthGenerator:
    """Class to generate ground truth from deterministic algorithm fitted data 
    for one run

    Inputs: 
        data, a nx2 numpy array where each row represents a shot from the run. 
        The first column should be the predicted delay from the deterministic 
        algorithm; the second column should be the offset recorded by the 
        detector (for run 74 this ranges from -4.00 to 4.00)

    Outputs: 
        call fitData() which will return an array of length four that represents
        coefficients for a third-degree polynomial. The poly will take a 
        deterministic-generated value for an image and output a corrected 
        detector offset that should allow us to get a time in picoseconds. Use 
        the poly as such:
            correctedOffset = np.power(deterministic, 3) * poly[0] \
              + np.power(deterministic, 2) * poly[1] \
              + np.power(deterministic, 1) * poly[2] + poly[3]

    Alternatively you can just call correctPoint on a determinstic value and it 
    will return the corrected offset.
    """
    def __init__(self, data):
        self.data = data
        self.polyFit = None

    def correctPoint(self, deterministicVal):
        if self.polyFit is None:
            print("No polyfit generated yet! Call fitData")
            return

        return np.power(deterministicVal, 3) * self.polyFit[0] \
          + np.power(deterministicVal, 2) * self.polyFit[1] \
          + np.power(deterministicVal, 1) * self.polyFit[2] + self.polyFit[3]

    def fitData(self):
        """** WORD OF WARNING **
        The paramaters for the ransac funtion are tuned for run 74 where there 
        are ~150 distinct xray offsets and ~40000 images. No guarantee about 
        what happens if you try with anything else :)
        """
        meanData = self.computeMeans()
        fit1 = self.ransac(meanData, 100, 20, 0.2, 25000)
        curveAdjustedData = self.adjustDataToCurve(fit1)
        fit2 = self.ransac(curveAdjustedData, 10000, 30, 0.75, 10000)
        self.plotRansac(fit2, curveAdjustedData, "plot.png")
        self.polyFit = fit2
        return fit2

    def computeMeans(self):
        """Find the mean phase slope for each detector offset. 

        We assume one detector offset will produce normally-distributed data.
        """
        uniqueX = np.unique(self.data[:,1])
        meanData = np.zeros((len(uniqueX), 2))

        for tx, xVal in enumerate(uniqueX):
            mean = np.mean(self.data[self.data[:,1] == xVal][:,0])
            meanData[tx,:] = [mean, xVal]

        return meanData

    def adjustDataToCurve(self, firstFit):
        """
        With the deterministic algorithm some points end up 2pi off where they 
        should be. This allows us to use the initial curve estimate to "correct"
        these points and put them back where they belong
        """
        dataCopy = np.copy(self.data)
        for row in range(dataCopy.shape[0]):

            polyToSolve = np.copy(firstFit);
            polyToSolve[3] -= dataCopy[row, 1];
            solve = np.roots(polyToSolve);

            fitGuess = 1e5
            for root in solve:
                if not np.iscomplex(root):
                    fitGuess = root


            originalGuess = dataCopy[row, 0]

            if abs(fitGuess - originalGuess - 2 * np.pi) < (np.pi / 2.0):
                dataCopy[row, 0] += 2 * np.pi

        return dataCopy

    def ransac(self, data, goodFitCount, numSample, inlierDelta, it):
        """
        Simple RANSAC implementation. The best fitting algorithm ever! This one
        assumes fitting a cubic
        """
        bestFitResid = 1e15
        bestCoef = []

        for i in range(it):
            idx = np.random.randint(data.shape[0], size=numSample)
            subset = data[idx,:]

            # Get a cubic fit for the data
            fit = np.polyfit(subset[:,0], subset[:,1], 3)

            # Evaluate the fit on all the phase slopes
            guess = np.power(
                data[:,0], 3) * fit[0] \
                + np.power(data[:,0], 2) * fit[1] \
                + np.power(data[:,0], 1) * fit[2] \
                + fit[3]

            # Calculate the residuals against the actual phase slopes
            resid = np.abs(data[:,1] - guess)
            
            # Count how many are inliers
            countInlier = len(resid[resid <= inlierDelta])
            
            # 
            if countInlier > goodFitCount:

                bigSubset = data[resid <= inlierDelta]
                bigFit = np.polyfit(bigSubset[:,0], bigSubset[:,1], 3)

                bigGuess = np.power(bigSubset[:,0], 3) * bigFit[0] \
                  + np.power(bigSubset[:,0], 2) * bigFit[1] \
                  + np.power(bigSubset[:,0], 1) * bigFit[2] + bigFit[3]

                totalResid = np.sum(np.abs(bigSubset[:,1] - bigGuess))

                if totalResid < bestFitResid:
                    bestCoef = bigFit
                    bestFitResid = totalResid
                    print(bestFitResid)

        return bestCoef

    def plotRansac(self, coef, data, plotfile):
        minX = np.amin(data[:,0])
        maxX = np.amax(data[:,0])
        xLine = np.linspace(minX, maxX, num=50)
        yLine = np.power(xLine, 3) * coef[0] + np.power(xLine, 2) * coef[1] \
          + np.power(xLine, 1) * coef[2] + coef[3]

        plt.switch_backend('agg');
        plt.plot(data[:,0], data[:,1], 'ro');
        plt.plot(xLine, yLine);
        plt.savefig(plotfile);
