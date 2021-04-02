import os
import sys
user = "priyabapat"
path = "/Users/" + user + "/Code/relaxation-labeling/core/"
sys.path.append(os.path.dirname(path))
print(sys.path)
from relax import RelaxationLabeling

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ortho_group

class IdentifyPoints(RelaxationLabeling):
    def __init__(self):
        super(IdentifyPoints, self).__init__()

    def initPointObjectsAndLabels(self, Dim, NumObjects, NumLabels, Noise, DeleteLabel, ObjectOffset, ObjectScale, ShuffleObjects, RotateObjects):
        labels = np.random.rand(NumLabels,Dim)

        objects = np.copy(labels)

        if RotateObjects:
            rotated = np.zeros(objects.shape)
            mat = ortho_group.rvs(dim=Dim)
            print('mat')
            print(mat)

            for i in range(0,objects.shape[0]):
                rotated[i] = np.matmul(mat, objects[i])
                print('i mag... object',np.linalg.norm(objects[i]),'  rotated ',np.linalg.norm(rotated[i]))
            print('objects',objects)
            print('rotated',rotated)
            objects = np.copy(rotated)

        if ShuffleObjects:
            np.random.shuffle(objects)

        if Noise > 0.0:
            noise = Noise*np.random.rand(objects.shape)
            objects += Noise

        if DeleteLabel >= 0:
            labels = np.delete(labels, DeleteLabel, axis=0)
            #objects = np.append(objects, [[50.14156, 70.7134]], axis=0)

        if ObjectOffset[0] > 0.0:
            objects += ObjectOffset

        if ObjectScale != 1.0:
            objects *= ObjectScale

        return objects, labels

    def init3DCompat(self, objects, labels):

        NumObjects = objects.shape[0]
        NumLabels = labels.shape[0]
        r = np.zeros((NumObjects, NumLabels, NumObjects, NumLabels, NumObjects, NumLabels))

        epsilon = 1e-8

        UseDistance = False
        UseAngle = True
        AngleTechnique = 3
        for i1 in range(0,NumObjects):
            for l1 in range(0,NumLabels):
                for i2 in range(0,NumObjects):
                    for l2 in range(0,NumLabels):
                        for i3 in range(0,NumObjects):
                            for l3 in range(0,NumLabels):
                                if False:
                                    vobj21 = objects[i1]-objects[i2]
                                    vobj23 = objects[i3]-objects[i2]
                                    vlab21 = labels[l1]-labels[l2]
                                    vlab23 = labels[l3]-labels[l2]
                                if True:
                                    vobj21 = objects[i2]-objects[i1]
                                    vobj23 = objects[i3]-objects[i1]
                                    vlab21 = labels[l2]-labels[l1]
                                    vlab23 = labels[l3]-labels[l1]
                                nvobj21 = np.linalg.norm(vobj21)
                                nvobj23 = np.linalg.norm(vobj23)
                                nvlab21 = np.linalg.norm(vlab21)
                                nvlab23 = np.linalg.norm(vlab23)
                                if nvobj21 > epsilon and nvobj23 > epsilon and nvlab21 > epsilon and nvlab23 > epsilon:
                                    vobj21 /= nvobj21
                                    vobj23 /= nvobj23
                                    vlab21 /= nvlab21
                                    vlab23 /= nvlab23
                                    r[i1,l1,i2,l2,i3,l3] = 1.0/(1.0+abs(np.dot(vobj21,vobj23) - np.dot(vlab21,vlab23)))

        p = np.zeros((NumObjects, NumLabels))
        for i in range(0,NumObjects):
            for j in range(0,NumLabels):
                p[i,j] = 1.0/NumLabels

        return r, p
                    
    def twoElementPermutations(self, Dim):
        perms = list()
        for i in range(0,Dim):
            for j in range(i+1,Dim):
                perms += [[i,j]]

        return perms

    def main(self):


        Dim = 28
        NumLabels = 10
        NumObjects = 10
        PlotPerms = self.twoElementPermutations(Dim)
        MaxNumPlots = 1
        print('perms ',PlotPerms)


        Noise = 0.0
        DeleteLabel = -1
        ObjectOffset = 3*np.ones((Dim))
        ShuffleObjects = False
        ObjectScale = 2.0
        RotateObjects = True
        CompatType = 3

        objects, labels =  self.initPointObjectsAndLabels(Dim, NumObjects, NumLabels, Noise, DeleteLabel, ObjectOffset, ObjectScale, ShuffleObjects, RotateObjects)

        if CompatType == 2:
            r, p = self.init2DCompat(objects, labels)
        if CompatType == 3:
            r, p = self.init3DCompat(objects, labels)


        numPlot = 0
        for indx,perm in enumerate(PlotPerms):
            plt.plot(labels[:,perm[0]], labels[:,perm[1]], 'ro')
            plt.plot(objects[:,perm[0]], objects[:,perm[1]], 'g+')
            for i in range(0,labels.shape[0]):
                plt.plot([labels[i,perm[0]], objects[i,perm[0]]], [labels[i,perm[1]], objects[i,perm[1]]], linewidth=1.0)
            plt.title('Objects Shuffled ' + str(perm))
            plt.show()
            if indx >= MaxNumPlots: break


        NumObjects = objects.shape[0]
        NumLabels = labels.shape[0]
        print('Num objects',NumObjects)
        print('Num labels',NumLabels)
        s = np.zeros((NumObjects, NumLabels))
        supportFactor = 1.0

        NumIterations = 30
        n = 0
        while n < NumIterations:
            self.updateSupport(CompatType, NumObjects, NumLabels, s, p, r)
            self.updateProbability(NumObjects, NumLabels, s, p, r, supportFactor)
            n += 1
        print('s', s)
        print('p', p)
        print('labeling from p')
        objectToLabelMapping = np.zeros((NumObjects,1))
        for i in range(0,NumObjects):
            jmax = np.argmax(p[i,:])
            objectToLabelMapping[i] = jmax
            #print('Obj#',i,'Label# ',jmax,' Obj ',objects[i],' Label ',labels[jmax], '  p ',p[i,jmax])
            print('Obj#',i,'Label# ',jmax,'  p ',p[i,jmax])
            if False:
                if np.linalg.norm(objects[i,:] - labels[jmax,:]) > 1e-4:
                    print('probs for object i',i)
                    print(p[i,:])

        for indx,perm in enumerate(PlotPerms):
            plt.plot(objects[:,perm[0]], objects[:,perm[1]], 'go')
            for i in range(0,NumObjects):
                j = int(objectToLabelMapping[i,0])
                plt.plot([labels[j,perm[0]]], [labels[j,perm[1]]], 'r+')
                plt.plot([labels[j,perm[0]], objects[i,perm[0]]], [labels[j,perm[1]], objects[i,perm[1]]], linewidth=1.0)
            plt.title('Objects Labeled '+str(perm))
            plt.show()
            if indx >= MaxNumPlots: break



identifyPoints = IdentifyPoints()
identifyPoints.main()
