import numpy as np

class RelaxationLabeling(object):

    def __init__(self, compatType, save):
        self.compatType = compatType # Are we dealing with two pairs of object-labels (2) or three pairs of object-labels (3)
        self.save = save
        self.supportFactor = 1.0
        self.iterations = 30
        self.iteration = 0

    def initStrengthAndSupport(self):
        self.strength = np.ones(shape = [self.numObjects, self.numLabels])*1/self.numLabels
        self.support = np.zeros(shape = [self.numObjects, self.numLabels])

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

    def updateStrength(self):
        technique = 2
        if technique == 1:
            for i in range(self.numObjects):
                for j in range(self.numLabels):
                    self.strength[i,j] += self.support[i,j]*self.supportFactor
                self.normalizeStrength(i)
        if technique == 2:
            for i in range(self.numObjects):
                den = 0.0
                for j in range(0,self.numLabels):
                    den += self.strength[i,j]*(1.0+self.support[i,j])
                for j in range(0,self.numLabels):
                    self.strength[i,j] = self.strength[i,j]*(1.0+self.support[i,j])/den
                self.normalizeStrength(i)

    def normalizeStrength(self, i):
        technique = 2
        if technique == 1 or technique == 2:
            minStrength = np.amin(self.strength[i, :])
            self.strength[i, :] -= minStrength
            if technique == 2:
                sumStrength = np.sum(self.strength[i, :])
                self.strength[i, :] /= sumStrength

    def iterate(self):
        print("iteration {}".format(self.iteration))
        self.updateSupport()
        self.updateStrength()
        self.iteration += 1

    def assign(self):
        print('labeling from strength')
        self.objectToLabelMapping = np.zeros((self.numObjects,1))
        for i in range(0,self.numObjects):
            jmax = np.argmax(self.strength[i,:])
            self.objectToLabelMapping[i] = jmax
            print('Obj#',i,' Label# ',jmax,'strength ',self.strength[i,jmax])
            if False:
                if np.linalg.norm(self.objects[i,:] - self.labels[jmax,:]) > 1e-4:
                    print('strengths for object i',i)
                    print(self.strength[i,:])

    def saveCompatibilityForPlotting(self, compatibilityFilename, compatibility):
        compatibilityFile = open(compatibilityFilename, 'w')
        compatibilityText = ''
        for i in range(self.numObjects):
            for j in range(self.numLabels):
                # One column in header row:
                compatibilityText += ',' + str(i) + str(j)  
        for i in range(self.numObjects):
            for j in range(self.numLabels):
                # One row in header column:
                compatibilityText += '\n' + str(i) + str(j) 
                for k in range(self.numObjects):
                    for l in range(self.numLabels):
                        # One compatibility value:
                        compatibilityText += ',' + str(compatibility[i,j,k,l])
        compatibilityFile.write(compatibilityText)
        compatibilityFile.close()
        
    def saveCompatibility(self, compatibilityFilename, compatibility):
        compatibilityFile = open(compatibilityFilename, 'w')
        compatibilityText = ''
        for i in range(self.numObjects):
            for j in range(self.numLabels):
                # One column in header row:
                compatibilityText += ',' + '[' + str(i) + ']' + '[' + str(j) + ']'
        for i in range(self.numObjects):
            for j in range(self.numLabels):
                # One row in header column:
                compatibilityText += '\n' + '[' + str(i) + ']' + '[' + str(j) + ']'
                for k in range(self.numObjects):
                    for l in range(self.numLabels):
                        # One compatibility value:
                        compatibilityText += ',' + str(compatibility[i,j,k,l])
        compatibilityFile.write(compatibilityText)
        compatibilityFile.close()

    def main(self):
        print("objects:")
        print(self.objects)
        print("labels:")
        print(self.labels)
        self.calculateCompatibility()
        self.initStrengthAndSupport()
        print('Num objects', self.numObjects)
        print('Num labels', self.numLabels)
        for i in range(self.iterations):
            self.iterate()
        print('support', self.support)
        print('strength', self.strength)
        self.assign()

