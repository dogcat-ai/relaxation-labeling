# Here, we want to use the relaxation labeling algorithm to verify the chessboard corners found with OpenCV's findChessboardCorners function.

import debug_tabs as dt
import os
import sys
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
        numLabels = 2 # "good" or "bad"
        numObjects = self.findChessboardCorners.nx*self.findChessboardCorners.ny
        compatType = 3
        save = True
        super(VerifyChessboardCorners, self).__init__(numLabels, numObjects, compatType, save)

    def calculateCompatibility(self):
        if self.compatType == 2:
            self.debugTabs.print("ERROR: Expected compatType == 3, but got 2")
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
                                    self.compatibility[i,j,k,l,m,n] = 

