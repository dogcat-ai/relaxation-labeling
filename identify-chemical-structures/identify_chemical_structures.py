import os
import sys
user = "priyabapat"
path = "/Users/" + user + "/Code/relaxation-labeling/core/"
sys.path.append(os.path.dirname(path))
print(sys.path)
from relax import RelaxationLabeling

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ortho_group

class IdentifyChemicalStructure(RelaxationLabeling):
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

        super(IdentifyChemicalStructure, self).__init__(dim,  numLabels, numObjects,  maxNumPlots, noise, deleteLabel, objectOffset, shuffleObjects, objectScale,  rotateObjects, compatType)

    def readImage(self):
        imagePath = path + "../../relaxation-labeling-supporting-files/single_bonds.jpeg"
        self.image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        self.imageColor = cv2.imread(imagePath, cv2.IMREAD_COLOR)
        print("After initial imread, image's shape is:")
        print("\t{}".format(self.image.shape))
        #(thresh, self.image) = cv2.threshold(self.image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        #print("After Otsu thresholding, image's shape is:")
        #print("\t{}".format(self.image.shape))

        cv2.imshow("molecule with single bonds", self.image)
        cv2.waitKey()

    def doEdgeDetection(self):
        self.edgeImage = cv2.Canny(self.image,50,150,apertureSize = 3)

    def doHoughLinesP(self):
        lines = cv2.HoughLinesP(self.edgeImage,1,np.pi/180,2,0,0)
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(self.imageColor,(x1,y1),(x2,y2),(0,0,255),2)

        cv2.imshow("image with hough lines", self.imageColor)
        cv2.waitKey()

    def doHoughLines(self):

        lines = cv2.HoughLines(self.edgeImage,1,np.pi/180,50)
        for line in lines:
            for rho,theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))

                cv2.line(self.imageColor,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.imshow("image with hough lines", self.imageColor)
        cv2.waitKey()

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

        return self.objects, self.labels


    def init3DCompat(self):

        self.r = np.zeros((self.numObjects, self.numLabels, self.numObjects, self.numLabels, self.numObjects, self.numLabels))

        epsilon = 1e-8

        useDistance = False
        useAngle = True
        angleTechnique = 3
        for i1 in range(0,self.numObjects):
            for l1 in range(0,self.numLabels):
                for i2 in range(0,self.numObjects):
                    for l2 in range(0,self.numLabels):
                        for i3 in range(0,self.numObjects):
                            for l3 in range(0,self.numLabels):
                                if False:
                                    vobj21 = self.objects[i1]-self.objects[i2]
                                    vobj23 = self.objects[i3]-self.objects[i2]
                                    vlab21 = self.labels[l1]-self.labels[l2]
                                    vlab23 = self.labels[l3]-self.labels[l2]
                                if True:
                                    vobj21 = self.objects[i2]-self.objects[i1]
                                    vobj23 = self.objects[i3]-self.objects[i1]
                                    vlab21 = self.labels[l2]-self.labels[l1]
                                    vlab23 = self.labels[l3]-self.labels[l1]
                                nvobj21 = np.linalg.norm(vobj21)
                                nvobj23 = np.linalg.norm(vobj23)
                                nvlab21 = np.linalg.norm(vlab21)
                                nvlab23 = np.linalg.norm(vlab23)
                                if nvobj21 > epsilon and nvobj23 > epsilon and nvlab21 > epsilon and nvlab23 > epsilon:
                                    vobj21 /= nvobj21
                                    vobj23 /= nvobj23
                                    vlab21 /= nvlab21
                                    vlab23 /= nvlab23
                                    self.r[i1,l1,i2,l2,i3,l3] = 1.0/(1.0+abs(np.dot(vobj21,vobj23) - np.dot(vlab21,vlab23)))

        self.p = np.zeros((self.numObjects, self.numLabels))
        for i in range(0,self.numObjects):
            for j in range(0,self.numLabels):
                self.p[i,j] = 1.0/self.numLabels

        return self.r, self.p
                    
    def twoElementPermutations(self, dim):
        perms = list()
        for i in range(0,dim):
            for j in range(i+1,dim):
                perms += [[i,j]]

        return perms

    def main(self):
        self.readImage()
        self.doEdgeDetection()
        self.doHoughLinesP()
        exit(0)
        self.objects, self.labels = self.initPointObjectsAndLabels()

        self.numObjects = self.objects.shape[0]
        self.numLabels = self.labels.shape[0]
        print('Num objects',self.numObjects)
        print('Num labels',self.numLabels)
        self.s = np.zeros((self.numObjects, self.numLabels))

        if self.compatType == 2:
            self.r, self.p = self.init2DCompat()
        if self.compatType == 3:
            self.r, self.p = self.init3DCompat()

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

        for i in range(self.iterations):
            self.iterate()

        print('s', self.s)
        print('p', self.p)
        print('labeling from p')
        objectToLabelMapping = np.zeros((self.numObjects,1))
        for i in range(0,self.numObjects):
            jmax = np.argmax(self.p[i,:])
            objectToLabelMapping[i] = jmax
            print('Obj#',i,'Label# ',jmax,'  p ',self.p[i,jmax])
            if False:
                if np.linalg.norm(self.objects[i,:] - self.labels[jmax,:]) > 1e-4:
                    print('probs for object i',i)
                    print(self.p[i,:])

        for indx,perm in enumerate(self.plotPerms):
            plt.plot(self.objects[:,perm[0]], self.objects[:,perm[1]], 'go')
            for i in range(0,self.numObjects):
                j = int(objectToLabelMapping[i,0])
                plt.plot([self.labels[j,perm[0]]], [self.labels[j,perm[1]]], 'r+')
                plt.plot([self.labels[j,perm[0]], self.objects[i,perm[0]]], [self.labels[j,perm[1]], self.objects[i,perm[1]]], linewidth=1.0)
            plt.title('Objects Labeled '+str(perm))
            plt.show()
            if indx >= self.maxNumPlots:
                break

identifyChemicalStructure = IdentifyChemicalStructure()
identifyChemicalStructure.main()
