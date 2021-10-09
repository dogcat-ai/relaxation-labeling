# Here, we want to use the relaxation labeling algorithm to verify the chessboard corners found with OpenCV's findChessboardCorners function.

import os
import numpy as np
import sys

import debug_tabs as dt
from find_chessboard_corners import FindChessboardCorners
currentPath = os.path.dirname(os.path.realpath(__file__))
corePath = os.path.join(currentPath, '../core')
corePath = os.path.realpath(corePath)
sys.path.append(corePath)
from relax import RelaxationLabeling

class VerifyChessboardCorners(RelaxationLabeling):
    def __init__(self):
        self.debugTabs = dt.DebugTabs()
        self.findChessboardCorners = FindChessboardCorners()
        self.debugTabs.print("self.findChessboardCorners.corners: {}".format(self.findChessboardCorners.corners))
        exit(1)
        self.numObjects = self.findChessboardCorners.nx*self.findChessboardCorners.ny
        self.numLabels = 2 # "good" or "bad"
        self.compatType = 3 # 1: i,j; 2: k,l; 3: m,n
        self.save = True
        self.initCornerObjectsAndGoodBadLabels()
        super(VerifyChessboardCorners, self).__init__(self.numObjects, self.numLabels, self.compatType, self.save)

    def initCornerObjectsAndGoodBadLabels(self):
        self.labels = [0, 1] # 0: default, "good"; 1: special case, "bad"
        self.objects = [[]] # TODO|MEG|OPTIMIZATION|SPEED: Consider turning into a NumPy array
        self.debugTabs.print("nx: {} ny: {}".format(self.findChessboardCorners.nx, self.findChessboardCorners.ny))
        self.debugTabs.print("corners: {}".format(self.findChessboardCorners.corners))
        for c, corner in enumerate(self.findChessboardCorners.corners):
            if c % self.findChessboardCorners.nx and c == 0:
                self.objects.append([])
            self.objects[-1].append(corner)
            self.debugTabs.print("c: {} corner: {}".format(c,corner))
            self.debugTabs.print("objects: {}".format(self.objects))

    def calculateCompatibility(self):
        if self.compatType == 2:
            self.debugTabs.print("ERROR: Expected compatType == 3, but got 2")
            exit(1)
        if self.compatType == 3:
            self.calculate3DCompatibility()

    def calculate3DCompatibility(self):
        self.compatibility = np.zeros((self.numObjects, self.numLabels, self.numObjects, self.numLabels, self.numObjects, self.numLabels))
        for i in range(self.numObjects):
            for j in range(self.numLabels):
                for k in range(self.numObjects):
                    for l in range(self.numLabels):
                        for m in range(self.numObjects):
                            for n in range(self.numLabels):
                                    #self.compatibility[i,j,k,l,m,n] = 
                                    self.debugTabs.print("calculate3DCompatibility not implemented yet.  Exiting...")
                                    exit(1)

    def main(self):
        super(VerifyChessboardCorners, self).main()

verifyChessboardCorners = VerifyChessboardCorners()
verifyChessboardCorners.main()
