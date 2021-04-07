import numpy as np
import random
from itertools import permutations

class RelaxationLabeling(object):

    # n-dimensional cartesian points -> n-dimensional cartesian points
    # n-dimensional cartesian points -> m-dimensional (projections) cartesian points
    # n-dimensional lists -> n-dimensional lists
    # n-dimensional lists -> m-dimensional lists
    # labeling of chemical molecules
    # compat speed up
    #   multiplication of vector (and or matrices) not just elements one at a time
    
    def __init__(self, dim, numLabels, numObjects, maxNumPlots, noise, deleteLabel, objectOffset, shuffleObjects, objectScale, rotateObjects, compatType):
        
        self.dim = dim
        self.numLabels = numLabels
        self.numObjects = numObjects
        self.maxNumPlots = maxNumPlots

        self.noise = noise
        self.deleteLabel = deleteLabel
        self.objectOffset = 3*np.ones((self.dim))
        self.shuffleObjects = shuffleObjects
        self.objectScale = objectScale
        self.rotateObjects = rotateObjects
        self.compatType = compatType
        self.supportFactor = 1.0
        self.iterations = 30
        self.iteration = 0

    def updateSupport(self):
        if self.compatType == 2:
            for i in range(self.numObjects):
                for j in range(self.numLabels):
                    self.support[i,j] = 0.0
                    for k in range(self.numObjects):
                        for l in range(self.numLabels):
                            self.support[i,j] += self.strength[k,l]*self.compatibility[i,j,k,l]
                self.normalizeSupport(i)

        if self.compatType == 3:
            for i in range(self.numObjects):
                for j in range(self.numLabels):
                    for k in range(self.numObjects):
                        for l in range(self.numLabels):
                            for m in range(self.numObjects):
                                for n in range(self.numLabels):
                                    self.support[i,j] += self.strength[k,l]*self.strength[m,n]*self.compatibility[i,j,k,l,m,n]
                self.normalizeSupport(i)

    def normalizeSupport(self, i):
        minimumSupport = np.amin(self.support[i,:])
        maximumSupport = np.amax(self.support[i,:])
        maximumSupport -= minimumSupport
        self.support[i, :] = (self.support[i, :] - minimumSupport)/maximumSupport

    def updateProbability(self):
        technique = 2
        if technique == 1:
            for i in range(self.numObjects):
                for j in range(self.numLabels):
                    self.strength[i,j] += self.support[i,j]*self.supportFactor
                self.normalizeProbability(i)
        if technique == 2:
            for i in range(self.numObjects):
                den = 0.0
                for j in range(0,self.numLabels):
                    den += self.strength[i,j]*(1.0+self.support[i,j])
                for j in range(0,self.numLabels):
                    self.strength[i,j] = self.strength[i,j]*(1.0+self.support[i,j])/den
                self.normalizeProbability(i)

    def normalizeProbability(self, i):
        technique = 2
        if technique == 1 or technique == 2:
            minProbability = np.amin(self.strength[i, :])
            self.strength[i, :] -= minProbability
            if technique == 2:
                sumProbability = np.sum(self.strength[i, :])
                self.strength[i, :] /= sumProbability

    def iterate(self):
        print("iteration {}".format(self.iteration))
        self.updateSupport()
        self.updateProbability()
        self.iteration += 1

