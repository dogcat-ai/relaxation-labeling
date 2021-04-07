import os
import sys
user = "mannyglover"
path = "/Users/" + user + "/Code/relaxation-labeling/core/"
sys.path.append(os.path.dirname(path))
print(sys.path)
from relax import RelaxationLabeling

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ortho_group

class MergeLineSegments(RelaxationLabeling):
    def __init__(self):
        dim = 28
        maxNumPlots = 1
        noise = 0.0
        deleteLabel = -1
        objectOffset = 3*np.ones((dim))
        shuffleObjects = False
        objectScale = 2.0
        rotateObjects = False
        compatType = 2
        save = True
        super(MergeLineSegments, self).__init__(dim, maxNumPlots, noise, deleteLabel, objectOffset, shuffleObjects, objectScale, rotateObjects, compatType, save)

    def readImage(self):
        imagePath = path + "../../relaxation-labeling-supporting-files/triangular-bond-w-1-offshoot.jpeg"
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
        #lines = cv2.HoughLinesP(self.edgeImage,1,np.pi/180,25,10,10)
        #self.lines = cv2.HoughLinesP(self.edgeImage,rho = 1,theta = 1*np.pi/180,threshold = 25,minLineLength = 10,maxLineGap = 10)
        self.lines = cv2.HoughLinesP(self.edgeImage,rho = 1,theta = 1*np.pi/180,threshold = 10,minLineLength = 5,maxLineGap = 10)
        for l, line in enumerate(self.lines):
            for x1, y1, x2, y2 in line:
                print("line #{}: {} {} {} {}".format(l, x1, y1, x2, y2))
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

    def initLineSegmentObjectsAndLabels(self):
        self.readImage()
        self.doEdgeDetection()
        self.doHoughLinesP()
        self.objects = np.zeros(shape = [len(self.lines), 4])
        for l, line in enumerate(self.lines):
            self.objects[l] = line[0]
        self.labels = self.objects
        self.numObjects = len(self.objects)
        self.numLabels = len(self.labels)

    def calculateOrientationCompatibility(self, i, k):
        iObject = self.objects[i]
        kObject = self.objects[k]
        self.orientationCompatibility[i, k, k, i] = np.dot(iObject, kObject)

    def calculateProximityCompatibility(self, i, k):
        iObject = self.objects[i]
        kObject = self.objects[k]
        distance12 = np.linalg.norm(iObject[0:2] - kObject[2:])
        distance11 = np.linalg.norm(iObject[0:2] - kObject[0:2])
        distance21 = np.linalg.norm(iObject[2:] - kObject[0:2])
        distance22 = np.linalg.norm(iObject[2:] - kObject[2:])
        shortest = min(distance12, distance11, distance21, distance22)
        self.proximityCompatibility[i, k, k, i] = 3 - shortest

    def calculateCompatibility(self):
        self.orientationCompatibility = np.zeros((self.numObjects, self.numLabels, self.numObjects, self.numLabels))
        self.proximityCompatibility = np.zeros((self.numObjects, self.numLabels, self.numObjects, self.numLabels))
        self.compatibility = np.zeros((self.numObjects, self.numLabels, self.numObjects, self.numLabels))
        for i, iObject in enumerate(self.objects):
            for j, jLabel in enumerate(self.labels):
                for k, kObject in enumerate(self.objects):
                    for l, lLabel in enumerate(self.labels):
                        self.calculateOrientationCompatibility(i, k)
                        self.calculateProximityCompatibility(i, k)
                        self.compatibility[i, j, k, l] = 0.5*self.orientationCompatibility[i, j, k, l] + 0.5*self.proximityCompatibility[i, j, k, l]

    def main(self):
        self.initLineSegmentObjectsAndLabels()
        super(MergeLineSegments, self).main()


mergeLineSegments = MergeLineSegments()
mergeLineSegments.main()
