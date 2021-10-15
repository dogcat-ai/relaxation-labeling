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
        #   another corner i if there is a corner k that is a next door neighbor
        #   of i and m is a next door neighbor of k
        #   and k is in between i and m
        #
        # Get next door neighbors, same row.
        #   What is the best way to organize this data structure?
        #       One way is to have a list of lists of lists, where
        #       the list is organized by row; i.e.,
        #       with a 10 columns x 7 rows chessboard:
        #          [[ 00,00 00,01, 00,02, ..., 00,09 ]
        #           [ 01,00 01,01, 01,02, ..., 01,09 ]
        #           [ ..,.. ..,.., ..,.., ..., ..,.. ]
        #           [ 00,00 00,01, 00,02, ..., 00,09 ]]
        #       In the table above, "wx,yz" should be interpreted as
        #       a place holder, for a list corresponding to
        #       the corner at "wx,yz".  This list will look like:
        #           [ i j ] or [i]
        #       where i j are the next door neighbors of corner "wx,yz"
        #       and there will be no j where corner "wx,yz" is at the 0th
        #       column or the last column.
        #   Now that we have the organization of the data structure
        #   articulated, let's describe how to compose/populate the data
        #   structure:
        #       First, we will create a list of lists of lists, where the
        #           outermost list is ny units long, the middle list is nx
        #           units long, and the innermost list is an empty list.
        #       Second, we will iterate through all the corners in our chessboard,
        #           using nx (number of columns) and ny (number of rows).
        #           Unless our corner is the 0th corner of its row or
        #           the last corner of its row, we will add the j-1th corner
        #           and the j+1th corner to the innermost list, where j is the
        #           the column of the "corner at hand".
        #           If the "corner at hand" is the 0th corner,
        #           we will only add the j+1th corner.
        #           If the "corner at hand" is the last corner,
        #           we will only add the j-1th corner.
        nextDoorNeighborsSameRow = []
        for i in range(self.findChessboardCorners.ny):
            nextDoorNeighborsSameRow.append([])
            for j in range(self.findChessboardCorners.nx):
                nextDoorNeighborsSameRow[-1].append([])
                if j == 0:
                    nextDoorNeighborsSameRow[i][j].append(j+1)
                elif j == self.findChessboardCorners.nx - 1:
                    nextDoorNeighborsSameRow[i][j].append(j-1)
                else:
                    nextDoorNeighborsSameRow[i][j].append(j+1)
                    nextDoorNeighborsSameRow[i][j].append(j-1)
        self.debugTabs.print("nextDoorNeighborSameRow:")
        self.debugTabs.print(nextDoorNeighborsSameRow)

    def defineNeighborRelationsBak(self):
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
        self.debugTabs.print("[AFTER DECLARATION/DEFINITION] nextDoorNeighborsSameRow: {}".format(self.nextDoorNeighborsSameRow))
        self.nextDoorNeighborsSameColumn = []
        self.nonNextDoorNeighborsSameRow = []
        self.nonNextDoorNeighborsSameColumn = []
        for r, row in enumerate(self.objects):
            self.nextDoorNeighborsSameRow.append([])
            self.debugTabs.print("[AFTER 1ST APPEND] nextDoorNeighborsSameRow: {} r: {}".format(self.nextDoorNeighborsSameRow, r))
            self.nextDoorNeighborsSameColumn.append([])
            self.nonNextDoorNeighborsSameRow.append([])
            self.nonNextDoorNeighborsSameColumn.append([])
            for c, column in enumerate(row):
                self.nextDoorNeighborsSameRow[-1].append([])
                self.debugTabs.print("[AFTER 2ND APPEND] nextDoorNeighborsSameRow: {} c: {}".format(self.nextDoorNeighborsSameRow, c))
                self.nextDoorNeighborsSameColumn[-1].append([])
                self.nonNextDoorNeighborsSameRow[-1].append([])
                self.nonNextDoorNeighborsSameColumn[-1].append([])
                for rr, rowAgain in enumerate(self.objects):
                    self.nextDoorNeighborsSameRow[r][c].append([])
                    self.debugTabs.print("[AFTER 3RD APPEND] nextDoorNeighborsSameRow: {} rr: {}".format(self.nextDoorNeighborsSameRow, rr))
                    self.nonNextDoorNeighborsSameRow[r][c].append([])
                    self.nextDoorNeighborsSameColumn[r][c].append([])
                    self.nonNextDoorNeighborsSameColumn[r][c].append([])
                    for cc, columnAgain in enumerate(rowAgain):
                        self.nextDoorNeighborsSameRow[r][c][rr].append([])
                        self.debugTabs.print("[AFTER 4TH APPEND] nextDoorNeighborsSameRow: {} cc: {}".format(self.nextDoorNeighborsSameRow, cc))
                        self.nonNextDoorNeighborsSameRow[r][c][rr].append([])
                        self.nextDoorNeighborsSameColumn[r][c][rr].append([])
                        self.nonNextDoorNeighborsSameColumn[r][c][rr].append([])
                        """
                        self.debugTabs.print("nextDoorNeighborsSameRow: {}".format(self.nextDoorNeighborsSameRow))
                        self.debugTabs.print("r: {} c: {} rr: {} cc: {}".format(r,c,rr,cc))
                        """
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
