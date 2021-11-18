from compatibility import *

class CompatibilityChessboard(Compatibility3Pairs):
    def __init__(self, numObjects, numLabels):
        super.__init__(numObjects, numLabels)
        self.compatabilityCount = np.zeros(shape=(numObjects,numLabels,numObjects,numLabels,numObjects,numLabels),dtype = np.int)
        self.calculate()

    def calculate(findChessboardCorners):
        for r in range(findChessboardCorners.numRows):
            for c in range(findChessboardCorners.numColumns):
                self.calculateOneCorner(r, c, numRows, numColumns)
                
        """
        for i in range(self.numObjects):
            for j in range(self.numLabels):
                for k in range(self.numObjects):
                    for l in range(self.numLabels):
                        for m in range(self.numObjects):
                            for n in range(self.numLabels):
                                self.compatability(i,j,k,l,m,n) /= self.compatibilityCount(i,j,k,l,m,n)
        """
        self.compatibility /= self.compatibilityCount

    def calculateOneCorner(r, c, numRows, numColumns):
        
        compatibilityValue = 0.0

        if (c+2 < numColumns):
            compatibilityValue += self.calculateOneDirection(r, c, r, c+1, r, c+2)
        if (c-2 < numColumns):
            compatibilityValue += self.calculateOneDirection(r, c, r, c-1, r, c-2)
        if (r-2 < numColumns):
            compatibilityValue += self.calculateOneDirection(r, c, r-1, c, r-2, c)
        if (r+2 < numColumns):
            compatibilityValue += self.calculateOneDirection(r, c, r+1, c, r+2, c)

    def calculateOneDirection(self, rowOrigin, columnOrigin, rowNear, columnNear, rowFar, columnFar):
        origin = np.array([findChessboardCorners.corners[rowOrigin][columnOrigin][0],findChessboardCorners.corners[rowOrigin][columnOrigin][1]])
        origin = np.array([findChessboardCorners.corners[rowOrigin][columnOrigin][0],findChessboardCorners.corners[rowOrigin][columnOrigin][1]])
        near = np.array([findChessboardCorners.corners[rowNear][columnNear][0],findChessboardCorners.corners[rowNear][columnNear][1]])
        far = np.array([findChessboardCorners.corners[rowFar][columnFar][0],findChessboardCorners.corners[rowFar][columnFar][1]])

        originToNear = near - origin
        nearToFar = far - near

        compatibilityValue = np.dot(originToNear, nearToFar)

        i=rowOrigin*self.numRows+columnOrigin
        k=rowNear*self.numRows+columnNear
        m=rowFar*self.numRows+columnFar

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
        self.compatibility[i,1,k,1,m,0] -= compatibilityValue
        self.compatibilityCount[i,1,k,1,m,0] += 1
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

