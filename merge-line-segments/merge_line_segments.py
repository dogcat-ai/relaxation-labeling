import os
import sys
#user = "mannyglover"
#path = "/Users/" + user + "/Code/relaxation-labeling/core/"
path = "~/Code/relaxation-labeling/core/"
sys.path.append(os.path.dirname(os.path.expanduser(path)))
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
        #imagePath = path + "../../relaxation-labeling-supporting-files/triangular-bond-w-1-offshoot.jpeg"
        imagePath = os.path.expanduser("~/Code/relaxation-labeling-supporting-files/single_bonds.jpeg")
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
        self.lines = cv2.HoughLinesP(self.edgeImage,rho = 1,theta = 1*np.pi/180,threshold = 55,minLineLength = 20,maxLineGap = 10)
        #self.lines = cv2.HoughLinesP(self.edgeImage,rho = 1,theta = 1*np.pi/180,threshold = 10,minLineLength = 5,maxLineGap = 10)
        for l, line in enumerate(self.lines):
            for x1, y1, x2, y2 in line:
                print("line #{}: {} {} {} {}".format(l, x1, y1, x2, y2))
                self.imageColorCopy = self.imageColor.copy()
                cv2.line(self.imageColorCopy,(x1,y1),(x2,y2),(0,0,255),2)
                cv2.imwrite("line{}.png".format(l), self.imageColorCopy)

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
        self.objectDistances = np.zeros(shape = [len(self.lines)])
        self.objectVectors = np.zeros(shape = [len(self.lines), 2])
        for l, line in enumerate(self.lines):
            self.objects[l] = line[0]
            self.objectVectors[l] = self.objects[1, 0:2] - self.objects[l, 2:]
            self.objectDistances[l] = np.linalg.norm(self.objectVectors[l])
        self.labels = self.objects
        self.labelDistances  = self.objectDistances
        self.labelVectors = self.objectVectors
        self.numObjects = len(self.objects)
        self.numLabels = len(self.labels)

    def calculateOrientationCompatibility(self, i, j, k, l):
        iVector = self.objectVectors[i]
        jVector = self.labelVectors[j]
        kVector = self.objectVectors[k]
        lVector = self.labelVectors[l]
        ijCompatibility = np.dot(iVector, jVector)/(self.objectDistances[i]*self.labelDistances[j])
        klCompatibility = np.dot(kVector, lVector)/(self.objectDistances[k]*self.labelDistances[l])
        self.orientationCompatibility[i, j, k, l] = 0.5*ijCompatibility + 0.5*klCompatibility

    def calculateOrientationCompatibilityVerbose(self, i, j, k, l):
        iObject = self.objects[i]
        jLabel = self.labels[j]
        kObject = self.objects[k]
        lLabel = self.labels[l]
        ijCompatibility = np.dot(iObject, jLabel)/(self.objectDistances[i]*self.labelDistances[j])
        ilCompatibility = np.dot(iObject, lLabel)/(self.objectDistances[i]*self.labelDistances[l])
        klCompatibility = np.dot(kObject, lLabel)/(self.objectDistances[k]*self.labelDistances[l])
        kjCompatibility = np.dot(kObject, jLabel)/(self.objectDistances[k]*self.labelDistances[j])
        # If object i is compatible with label j and object k is compatible with label j,
        # that is evidence that object i and object k belong to the same line segment.
        # So compatibility(i, j, k, j) should be high.
        # One way to express this mathematically is compatibility(i, j, k, j) = 0.5*ijCompatibility + 0.5*kjCompatibility
        # If object i is compatible with label j and object k is compatible with label l,
        # that is evidence that object i should have label j and object k should have label l, but not necessarily
        # that object i and object k go together.
        # If object i is compatible with label l and object k is compatible with label l,
        # that is evidence that object i and object k belong to the same line segment.
        # So compatibility(i, l, k, l) should be high.
        # One way to express this mathematically is compatibility(i, l, k, l) = 0.5*ilCompatibility + 0.5*klCompatibility
        self.compatibility[i, j, k, j] = 0.5*ijCompatibility + 0.5*kjCompatibility
        self.compatibility[i, l, k, l] = 0.5*ilCompatibility + 0.5*klCompatibility
        self.compatibility[i, j, k, l] = 0.5*ijCompatibility + 0.5*klCompatibility
        self.compatibility[i, l, k, j] = 0.5*ilCompatibility + 0.5*kjCompatibility
        self.compatibility[i, j, i, j] = ijCompatibility
        # If object i is compatible with label j and object k is compatible with label l,
        # that tells us that i could be j, and k could be l,
        # but what does it tell us about the compatibility of i being j AND k being l?
        # Does it tell us anything?
        # I suppose it at least tells us that this assignment is reasonable,
        # but I fear it will counter the merging we are trying to accomplish.
        # In my illustration, I say that compatibility(1, 1, 2, 2) should be low,
        # because we want 2 to belong to 1, or 1 to belong to 2,
        # but we don't want them to be independent.
        # But I don't think we want this compatibility to be negative.
        # Perhaps it should be zero?
        # We could express this mathematically with compatibility(i, j, k, l) = ijCompatibility - klCompatibility.
        # What does the above equation imply?  If ijCompatibility is high and klCompatibility is high,
        # the total compatibility is close to zero.
        # If ijCompatibility is high and klCompatibility is low,
        # the total compatibility is high.
        # But do we want this?
        # I think we do want this.
        # Because what it means is that i likes j, and k doesn't like l.
        # No, we don't want this.
        # If i likes j and k doesn't like l,
        # then the compatibility of (i,j,k,l) should be halfway between good and bad.
        # Mathematically, compatibility(i, j, k, l) = 0.5*ijCompatibility + 0.5*klCompatibility


    def calculateProximityCompatibility(self, i, j, k, l):
        # If the labels are identical, we want the objects to be close.
        # Otherwise, we want the objects to be close only if the labels are close.
        # We can express this mathematically like:
        #   if (j == l):
        #       compatibility = Constant - minDist(i, k)
        #   else:
        #       compatibility(i, j, k, l) = minDist(i, k) - minDist(j, l)
        # or:
        #       compatibility(i, j, k, l) = minDist(i, k)*minDist(j, l)?

        iObject = self.objects[i]
        kObject = self.objects[k]
        distance12 = np.linalg.norm(iObject[0:2] - kObject[2:])
        distance11 = np.linalg.norm(iObject[0:2] - kObject[0:2])
        distance21 = np.linalg.norm(iObject[2:] - kObject[0:2])
        distance22 = np.linalg.norm(iObject[2:] - kObject[2:])
        ikShortest = min(distance12, distance11, distance21, distance22)
        if j == l:
            self.proximityCompatibility[i, j, k, l] = 3 - ikShortest
        else:
            jLabel = self.labels[j]
            lLabel = self.labels[l]
            distance12 = np.linalg.norm(jLabel[0:2] - lLabel[2:])
            distance11 = np.linalg.norm(jLabel[0:2] - lLabel[0:2])
            distance21 = np.linalg.norm(jLabel[2:] - lLabel[0:2])
            distance22 = np.linalg.norm(jLabel[2:] - lLabel[2:])
            jlShortest = min(distance12, distance11, distance21, distance22)
            self.proximityCompatibility[i, j, k, l] = abs(ikShortest - jlShortest)

    def calculateCompatibility(self):
        self.orientationCompatibility = np.zeros((self.numObjects, self.numLabels, self.numObjects, self.numLabels))
        self.proximityCompatibility = np.zeros((self.numObjects, self.numLabels, self.numObjects, self.numLabels))
        self.compatibility = np.zeros((self.numObjects, self.numLabels, self.numObjects, self.numLabels))
        for i, iObject in enumerate(self.objects):
            for j, jLabel in enumerate(self.labels):
                for k, kObject in enumerate(self.objects):
                    for l, lLabel in enumerate(self.labels):
                        self.calculateOrientationCompatibility(i, j, k, l)
                        self.calculateProximityCompatibility(i, j, k, l)
                        self.compatibility[i, j, k, l] = 0.5*self.orientationCompatibility[i, j, k, l] + 0.5*self.proximityCompatibility[i, j, k, l]

    def main(self):
        self.initLineSegmentObjectsAndLabels()
        super(MergeLineSegments, self).main()
        self.saveCompatibilityForPlotting("orientation_compatibility_plt.csv",self.orientationCompatibility)
        self.saveCompatibilityForPlotting("proximity_compatibility_plt.csv",self.proximityCompatibility)
        self.saveCompatibilityForPlotting("compatibility_plt.csv",self.compatibility)
        self.saveCompatibility("orientation_compatibility.csv",self.orientationCompatibility)
        self.saveCompatibility("proximity_compatibility.csv",self.proximityCompatibility)
        self.saveCompatibility("compatibility.csv",self.compatibility)


mergeLineSegments = MergeLineSegments()
mergeLineSegments.main()
