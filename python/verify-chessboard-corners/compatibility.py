import numpy as np

class Compatibility:
    def __init__(self, numObjects, numLabels):
        self.numObjects = numObjects
        self.numLabels = numLabels
        self.save = True
        self.verbose = 2

class Compatibility2Pairs(Compatibility):
    def __init__(self, numObjects, numLabels):
        self.__super__(numObjects, numLabels)
        self.compatibility = np.zeros(shape = (numObjects,numLabels,numObjects,numLabels), dtype = np.float64)

class Compatibility3Pairs(Compatibility):
    def __init__(self, numObjects, numLabels):
        super(Compatibility3Pairs,self).__init__(numObjects, numLabels)
        self.compatibility = np.zeros(shape = (numObjects,numLabels,numObjects,numLabels,numObjects,numLabels), dtype = np.float64)
