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
    def updateSupport(self, CompatType, NumObjects, NumLabels, s, p, r):
        if CompatType == 2:
            for i in range(0,NumObjects):
                for l in range(0, NumLabels):
                    s[i,l] = 0.0
                    for j in range(0,NumObjects):
                        for lp in range(0,NumLabels):
                            s[i,l] += p[j,lp]*r[i,l,j,lp]
                self.normalizeSupport(i, s)

        if CompatType == 3:
            for i1 in range(0,NumObjects):
                for l1 in range(0, NumLabels):
                    for i2 in range(0,NumObjects):
                        for l2 in range(0,NumLabels):
                            for i3 in range(0,NumObjects):
                                for l3 in range(0,NumLabels):
                                    s[i1,l1] += p[i2,l2]*p[i3,l3]*r[i1,l1,i2,l2,i3,l3]

                self.normalizeSupport(i1, s)


    def normalizeSupport(self, i, s):
        minimumSupport = np.amin(s[i,:])
        maximumSupport = np.amax(s[i,:])
        maximumSupport -= minimumSupport
        s[i, :] = (s[i, :] - minimumSupport)/maximumSupport


    def updateProbability(self, NumObjects, NumLabels, s, p, r, SupportFactor):
        Technique = 2
        if Technique == 1:
            for i in range(0,NumObjects):
                for l in range(0,NumLabels):
                    p[i,l] += s[i,l]*SupportFactor
                self.normalizeProbability(i, p)
        if Technique == 2:
            for i in range(0,NumObjects):
                den = 0.0
                for lp in range(0,NumLabels):
                    den += p[i,lp]*(1.0+s[i,lp])
                for l in range(0,NumLabels):
                    p[i,l] = p[i,l]*(1.0+s[i,l])/den
                self.normalizeProbability(i, p)


    def normalizeProbability(self, i, p):
        Technique = 2
        if Technique == 1 or Technique == 2:
            minProbability = np.amin(p[i, :])
            p[i, :] -= minProbability
            if Technique == 2:
                sumProbability = np.sum(p[i, :])
                p[i, :] /= sumProbability
                #sumProbability = np.sum(p[i, :])
                #if abs(sumProbability - 1.0) > 1e-6:
                #    print(' probabiility error sumProbability = ',sumProbability)
                #    exit()

