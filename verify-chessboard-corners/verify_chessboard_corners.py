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
        self.numObjects = self.findChessboardCorners.nx*self.findChessboardCorners.ny
        self.numLabels = 2 # "good" or "bad"
        self.compatType = 3 # 1: i,j; 2: k,l; 3: m,n
        self.save = True
        self.initCornerObjectsAndGoodBadLabels()
        super(VerifyChessboardCorners, self).__init__(self.numObjects, self.numLabels, self.compatType, self.save)
        self.main()

    def defineNeighborRelations(self):
        # We say that a corner k is a "next door neighbor" of another corner i
        #   if it is on the same row and only one unit away
        #   (i.e., abs(column(k) - column(i)) = 1),
        #   or the "converse": a corner k is a "next door neighbor" of another
        #   corner i if it is on the same column and only one unit away
        #   (i.e., abs(row(k) - row(i)) = 1).
        # We say that a corner m is a "non next door neighbor" of
        # another corner i if there is a corner k that is a next door neighbor
        # of i and m is a next door neighbor of k
        # and k is in between i and m
        #
        # Initialize neighbor relations lists:
        self.nextDoorNeighborsSameRow = []
        self.nextDoorNeighborsSameColumn = []
        self.nonNextDoorNeighborsSameRow = []
        self.nonNextDoorNeighborsSameColumn = []
        for r, row in enumerate(self.objects):
            self.nextDoorNeighborsSameRow.append([])
            self.nextDoorNeighborsSameColumn.append([])
            self.nonNextDoorNeighborsSameRow.append([])
            self.nonNextDoorNeighborsSameColumn.append([])
            for c, column in enumerate(row):
                self.nextDoorNeighborsSameRow[-1].append([])
                self.nextDoorNeighborsSameColumn[-1].append([])
                self.nonNextDoorNeighborsSameRow[-1].append([])
                self.nonNextDoorNeighborsSameColumn[-1].append([])
                for rr, rowAgain in enumerate(self.objects):
                    self.nextDoorNeighborsSameRow[r][c].append([])
                    self.nonNextDoorNeighborsSameRow[r][c].append([])
                    self.nextDoorNeighborsSameColumn[r][c].append([])
                    self.nonNextDoorNeighborsSameColumn[r][c].append([])
                    for cc, columnAgain in enumerate(rowAgain):
                        self.nextDoorNeighborsSameRow[r][c][rr].append([])
                        self.nonNextDoorNeighborsSameRow[r][c][rr].append([])
                        self.nextDoorNeighborsSameColumn[r][c][rr].append([])
                        self.nonNextDoorNeighborsSameColumn[r][c][rr].append([])
                        self.debugTabs.print("nextDoorNeighborsSameRow: {}".format(self.nextDoorNeighborsSameRow))
                        self.debugTabs.print("r: {} c: {} rr: {} cc: {}".format(r,c,rr,cc))
                        self.nextDoorNeighborsSameRow[r][c][cc][rr] = False
                        self.nextDoorNeighborsSameColumn[r][c][cc][rr] = False
                        self.nonNextDoorNeighborsSameRow[r][c][cc][rr] = False
                        self.nonNextDoorNeighborsSameColumn[r][c][cc][rr] = False
        # Populate neighbor relations lists
        for r, row in enumerate(self.objects):
            for c, column in enumerate(row):
                for rr, rowAgain in enumerate(self.objects):
                    for cc, columnAgain in enumerate(rowAgain):
                        if r == rr and abs(c - cc) == 1:
                            self.nextDoorNeighborsSameRow[r][c][rr][cc] = True
                        if r == rr and abs(c - cc) == 2:
                            self.nonNextDoorNeighborsSameRow[r][c][rr][cc] = True
                        if c == cc and abs(r - rr) == 1:
                            self.nextDoorNeighborsSameColumn[r][c][rr][cc] = True
                        if c == cc and abs(r - rr) == 2:
                            self.nonNextDoorNeighborsSameColumn[r][c][rr][cc] = True
                        self.debugTabs.print("nextDoorNeighborsSameRow[{}][{}][{}][{}] = {}".format(r,c,rr,cc,self.nextDoorNeighborsSameRow[r][c][rr][cc]))
                        self.debugTabs.print("nextDoorNeighborsSameColumn[{}][{}][{}][{}] = {}".format(r,c,rr,cc,self.nextDoorNeighborsSameColumn[r][c][rr][cc]))
                        self.debugTabs.print("nonNextDoorNeighborsSameRow[{}][{}][{}][{}] = {}".format(r,c,rr,cc,self.nonNextDoorNeighborsSameRow[r][c][rr][cc]))
                        self.debugTabs.print("nonNextDoorNeighborsSameColumn[{}][{}][{}][{}] = {}".format(r,c,rr,cc,self.nonNextDoorNeighborsSameColumn[r][c][rr][cc]))

    def initCornerObjectsAndGoodBadLabels(self):
        self.labels = [0, 1] # 0: default, "good"; 1: special case, "bad"
        self.objects = [] # TODO|MEG|OPTIMIZATION|SPEED: Consider turning into a NumPy array
        self.debugTabs.print("nx: {} ny: {}".format(self.findChessboardCorners.nx, self.findChessboardCorners.ny))
        self.debugTabs.print("corners: {}".format(self.findChessboardCorners.corners))
        for r in range(self.findChessboardCorners.ny):
            self.objects.append([])
            for c in range(self.findChessboardCorners.nx):
                i = self.findChessboardCorners.nx*r + c
                self.objects[-1].append(self.findChessboardCorners.corners[i])
        self.debugTabs.print("objects: {}".format(self.objects))
        self.debugTabs.print("len(objects): {}".format(len(self.objects)))
        self.debugTabs.print("len(objects[0]): {}".format(len(self.objects[0])))
        self.debugTabs.print("objects, one row at a time:")
        for r, row in enumerate(self.objects):
            self.debugTabs.print("r: {} row: {}".format(r, row))
        self.defineNeighborRelations()

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
