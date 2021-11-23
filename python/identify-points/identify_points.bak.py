import os
import sys
currentPath = os.path.dirname(os.path.realpath(__file__))
corePath = os.path.join(currentPath, '../core')
corePath = os.path.realpath(corePath)
sys.path.append(corePath)

from relax import RelaxationLabeling

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ortho_group

class IdentifyPoints(RelaxationLabeling):
    def __init__(self):
        dim = 28
        numLabels = 10
        numObjects = 10
        maxNumPlots = 1

        noise = 0.0
        deleteLabel = -1
        objectOffset = 3*np.ones((dim))
        shuffleObjects = False
        objectScale = 2.0
        rotateObjects = True
        compatType = 3
        save = True

        super(IdentifyPoints, self).__init__(dim,  numLabels, numObjects,  maxNumPlots, noise, deleteLabel, objectOffset, shuffleObjects, objectScale,  rotateObjects, compatType, save)
        self.doPermutations = True

    def initPointObjectsAndLabels(self):
        self.labels = np.random.rand(self.numLabels,self.dim)
        self.objects = np.copy(self.labels)

        if self.rotateObjects:
            rotated = np.zeros(self.objects.shape)
            mat = ortho_group.rvs(dim=self.dim)
            print('mat')
            print(mat)

            for i in range(0,self.objects.shape[0]):
                rotated[i] = np.matmul(mat, self.objects[i])
                print('i mag... object',np.linalg.norm(self.objects[i]),'  rotated ',np.linalg.norm(rotated[i]))
            print('objects',self.objects)
            print('rotated',rotated)
            self.objects = np.copy(rotated)

        if self.shuffleObjects:
            np.random.shuffle(self.objects)

        if self.noise > 0.0:
            noise = noise*np.random.rand(self.objects.shape)
            self.objects += self.noise

        if self.deleteLabel >= 0:
            self.labels = np.delete(self.labels, self.deleteLabel, axis=0)

        if self.objectOffset[0] > 0.0:
            self.objects += self.objectOffset

        if self.objectScale != 1.0:
            self.objects *= self.objectScale

        self.numObjects = len(self.objects)
        self.numLabels = len(self.labels)


    def calculateCompatibility(self):
        if self.compatType == 2:
            self.calculate2DCompatibility()
        if self.compatType == 3:
            self.calculate3DCompatibility()

    def calculate3DCompatibility(self):
        self.compatibility = np.zeros((self.numObjects, self.numLabels, self.numObjects, self.numLabels, self.numObjects, self.numLabels))
        epsilon = 1e-8
        useDistance = False
        useAngle = True
        angleTechnique = 3
        for i in range(self.numObjects):
            for j in range(self.numLabels):
                for k in range(self.numObjects):
                    for l in range(self.numLabels):
                        for m in range(self.numObjects):
                            for n in range(self.numLabels):
                                if False:
                                    vobj21 = self.objects[i]-self.objects[k]
                                    vobj23 = self.objects[m]-self.objects[k]
                                    vlab21 = self.labels[j]-self.labels[l]
                                    vlab23 = self.labels[n]-self.labels[l]
                                if True:
                                    vobj21 = self.objects[k]-self.objects[i]
                                    vobj23 = self.objects[m]-self.objects[i]
                                    vlab21 = self.labels[l]-self.labels[j]
                                    vlab23 = self.labels[n]-self.labels[j]
                                nvobj21 = np.linalg.norm(vobj21)
                                nvobj23 = np.linalg.norm(vobj23)
                                nvlab21 = np.linalg.norm(vlab21)
                                nvlab23 = np.linalg.norm(vlab23)
                                if nvobj21 > epsilon and nvobj23 > epsilon and nvlab21 > epsilon and nvlab23 > epsilon:
                                    vobj21 /= nvobj21
                                    vobj23 /= nvobj23
                                    vlab21 /= nvlab21
                                    vlab23 /= nvlab23
                                    self.compatibility[i,j,k,l,m,n] = 1.0/(1.0+abs(np.dot(vobj21,vobj23) - np.dot(vlab21,vlab23)))

    def twoElementPermutations(self, dim):
        perms = list()
        for i in range(0,dim):
            for j in range(i+1,dim):
                perms += [[i,j]]

        return perms

    def permutatePre(self):
        self.plotPerms = self.twoElementPermutations(self.dim)
        print('perms ',self.plotPerms)
        self.numPlot = 0
        for indx,perm in enumerate(self.plotPerms):
            plt.plot(self.labels[:,perm[0]], self.labels[:,perm[1]], 'ro')
            plt.plot(self.objects[:,perm[0]], self.objects[:,perm[1]], 'g+')
            for i in range(0,self.labels.shape[0]):
                plt.plot([self.labels[i,perm[0]], self.objects[i,perm[0]]], [self.labels[i,perm[1]], self.objects[i,perm[1]]], linewidth=1.0)
            plt.title('Objects Shuffled ' + str(perm))
            plt.show()
            if indx >= self.maxNumPlots:
                break

    def permutatePost(self):
        for indx,perm in enumerate(self.plotPerms):
            plt.plot(self.objects[:,perm[0]], self.objects[:,perm[1]], 'go')
            for i in range(0,self.numObjects):
                j = int(self.objectToLabelMapping[i,0])
                plt.plot([self.labels[j,perm[0]]], [self.labels[j,perm[1]]], 'r+')
                plt.plot([self.labels[j,perm[0]], self.objects[i,perm[0]]], [self.labels[j,perm[1]], self.objects[i,perm[1]]], linewidth=1.0)
            plt.title('Objects Labeled '+str(perm))
            plt.show()
            if indx >= self.maxNumPlots:
                break

    def main(self):
        self.initPointObjectsAndLabels()

        if self.doPermutations:
            self.permutatePre()

        super(IdentifyPoints, self).main()

        if self.doPermutations:
            self.permutatePost()


identifyPoints = IdentifyPoints()
identifyPoints.main()
