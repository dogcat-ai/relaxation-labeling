from compatibility import *
import find_chessboard_corners as fcc

class CompatibilityChessboard(Compatibility3Pairs):
    def __init__(self, findChessboardCorners):
        self.findChessboardCorners = findChessboardCorners
        self.numRows = self.findChessboardCorners.numRows
        self.numColumns = self.findChessboardCorners.numColumns
        self.numObjects = self.findChessboardCorners.numRows*self.findChessboardCorners.numColumns
        self.numLabels = 2
        super(CompatibilityChessboard, self).__init__(self.numObjects, self.numLabels)
        self.compatibilityCount = np.zeros(shape=(self.numObjects,self.numLabels,self.numObjects,self.numLabels,self.numObjects,self.numLabels),dtype = np.int)
        self.calculate()

    def calculate(self):
        for r in range(self.numRows):
            for c in range(self.numColumns):
                self.calculateOneCorner(r, c)
                
        """
        for i in range(self.numObjects):
            for j in range(self.numLabels):
                for k in range(self.numObjects):
                    for l in range(self.numLabels):
                        for m in range(self.numObjects):
                            for n in range(self.numLabels):
                                self.compatability(i,j,k,l,m,n) /= self.compatibilityCount(i,j,k,l,m,n)
        """
        print ('shapes ',self.compatibility.shape,self.compatibilityCount.shape)
        print (' min count ', np.amin(self.compatibilityCount), ' max ',np.amax(self.compatibilityCount))
        # TODO Should this be used in some form??
        #self.compatibility /= self.compatibilityCount

    def calculateOneCorner(self,r, c):
        
        if (c+2 < self.numColumns):
            self.calculateOneDirection(r, c, r, c+1, r, c+2)
        if (c-2 >= 0):
            self.calculateOneDirection(r, c, r, c-1, r, c-2)
        if (r+2 < self.numRows):
            self.calculateOneDirection(r, c, r+1, c, r+2, c)
        if (r-2 >= 0):
            self.calculateOneDirection(r, c, r-1, c, r-2, c)

    def calculateOneDirection(self, rowOrigin, columnOrigin, rowNear, columnNear, rowFar, columnFar):
        print ('Shape of  corners ',self.findChessboardCorners.corners.shape)
        print ('Num rows ',self.numRows,'  cols ',self.numColumns)
        origin = np.array([self.findChessboardCorners.corners[rowOrigin*self.numColumns+columnOrigin][0],self.findChessboardCorners.corners[rowOrigin*self.numColumns+columnOrigin][1]])
        near = np.array([self.findChessboardCorners.corners[rowNear*self.numColumns+columnNear][0],self.findChessboardCorners.corners[rowNear*self.numColumns+columnNear][1]])
        far = np.array([self.findChessboardCorners.corners[rowFar*self.numColumns+columnFar][0],self.findChessboardCorners.corners[rowFar*self.numColumns+columnFar][1]])

        originToNear = near - origin
        nearToFar = far - near
        
        originToNear /= np.linalg.norm(originToNear)
        nearToFar /= np.linalg.norm(nearToFar)

        compatibilityValue = np.dot(originToNear, nearToFar)

        # uT --> upperThreshold
        # dT --> threshold delta
        # dT = 1.0 - uT
        # If input compatibility is from uT to 1.0
        #   the output compatibility is from 0.0 to 1.0
        # Elseif the input compatibility is from (uT - dT) to uT
        #   the output compatibility is from 0.0 to -1
        # Else
        #   the output compatibility is -1

            
        uT = 0.9
        dT = 1.0-uT
        if compatibilityValue >= uT:
            compatibilityValue = (compatibilityValue-uT)/dT
        elif compatibilityValue >= uT - dT:
            compatibilityValue = -(1.0-((compatibilityValue - (uT - dT))/dT))
        else:
            compatibilityValue = -1

        i=rowOrigin*self.numColumns+columnOrigin
        k=rowNear*self.numColumns+columnNear
        m=rowFar*self.numColumns+columnFar

        self.compatibility[i,0,k,0,m,0] += compatibilityValue
        self.compatibilityCount[i,0,k,0,m,0] += 1
        self.compatibility[i,1,k,0,m,0] -= compatibilityValue
        self.compatibilityCount[i,1,k,0,m,0] += 1
        self.compatibility[i,0,k,1,m,0] -= compatibilityValue
        self.compatibilityCount[i,0,k,1,m,0] += 1
        self.compatibility[i,0,k,0,m,1] -= compatibilityValue
        self.compatibilityCount[i,0,k,0,m,1] += 1
        self.compatibility[i,1,k,1,m,0] -= compatibilityValue
        self.compatibilityCount[i,1,k,1,m,0] += 1
        self.compatibility[i,1,k,0,m,1] -= compatibilityValue
        self.compatibilityCount[i,1,k,0,m,1] += 1
        self.compatibility[i,0,k,1,m,1] -= compatibilityValue
        self.compatibilityCount[i,0,k,1,m,1] += 1
        self.compatibility[i,1,k,1,m,1] -= compatibilityValue
        self.compatibilityCount[i,1,k,1,m,1] += 1

    """
    def calculate(self):
        for i in range(self.numObjects):
            for j in range(self.numLabels):
                for k in range(self.numObjects):
                    for l in range(self.numLabels):
                        for m in range(self.numObjects):
                            for n in range(self.numLabels):
    """

