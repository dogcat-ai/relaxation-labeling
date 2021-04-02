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
            for i in range(0,self.numObjects):
                for l in range(0, self.numLabels):
                    self.s[i,l] = 0.0
                    for j in range(0,self.numObjects):
                        for lp in range(0,self.numLabels):
                            self.s[i,l] += self.p[j,lp]*self.r[i,l,j,lp]
                self.normalizeSupport(i)

        if self.compatType == 3:
            for i1 in range(0,self.numObjects):
                for l1 in range(0, self.numLabels):
                    for i2 in range(0,self.numObjects):
                        for l2 in range(0,self.numLabels):
                            for i3 in range(0,self.numObjects):
                                for l3 in range(0,self.numLabels):
                                    self.s[i1,l1] += self.p[i2,l2]*self.p[i3,l3]*self.r[i1,l1,i2,l2,i3,l3]
                self.normalizeSupport(i1)

    def normalizeSupport(self, i):
        minimumSupport = np.amin(self.s[i,:])
        maximumSupport = np.amax(self.s[i,:])
        maximumSupport -= minimumSupport
        self.s[i, :] = (self.s[i, :] - minimumSupport)/maximumSupport

    def updateProbability(self):
        technique = 2
        if technique == 1:
            for i in range(0,self.numObjects):
                for l in range(0,self.numLabels):
                    self.p[i,l] += self.s[i,l]*self.supportFactor
                self.normalizeProbability(i)
        if technique == 2:
            for i in range(0,self.numObjects):
                den = 0.0
                for lp in range(0,self.numLabels):
                    den += self.p[i,lp]*(1.0+self.s[i,lp])
                for l in range(0,self.numLabels):
                    self.p[i,l] = self.p[i,l]*(1.0+self.s[i,l])/den
                self.normalizeProbability(i)

    def normalizeProbability(self, i):
        technique = 2
        if technique == 1 or technique == 2:
            minProbability = np.amin(self.p[i, :])
            self.p[i, :] -= minProbability
            if technique == 2:
                sumProbability = np.sum(self.p[i, :])
                self.p[i, :] /= sumProbability

    def iterate(self):
        self.updateSupport()
        self.updateProbability()

