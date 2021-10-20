import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import debug_tabs as dt

class FindChessboardCorners:
    def __init__(self):
        self.debugTabs = dt.DebugTabs()
        self.verbose = 1
        self.useColorImage = True
        self.showOriginalImage = True
        self.doLargerCircles = False
        self.corners = []

        # Make a list of calibration images
        imageCase = 1
        if imageCase == 1:
            self.chessboardImagesNames = glob.glob('../../relaxation-labeling-supporting-files/verify-chessboard-corners/camera_cal/*.jpg')
            self.nx = 10
            self.ny = 7
        elif imageCase == 2:
            self.chessboardImagesNames = glob.glob('../../relaxation-labeling-supporting-files/stereo_rig/*.png')
            self.nx = 6
            self.ny = 5
        elif imageCase == 3:
            self.chessboardImagesNames = glob.glob('../../relaxation-labeling-supporting-files/calib_example1/*.tif')
            self.nx = 13
            self.ny = 12

        if self.verbose > 0:
            self.debugTabs.print("chessboard images names: {}".format(self.chessboardImagesNames))
            self.debugTabs.print("num columns: {}".format(self.nx))
            self.debugTabs.print("num rows: {}".format(self.ny))

        self.main()

    def convert2uint8(self, image):
        if image.dtype != np.uint8:
            # Scale to 0 to 255
            imin = np.amin(image)
            imax = np.amax(image)
            image = 255*((image - imin)/(imax-imin))
            # Round
            image = np.around(image)
            # Convert to uint8 type
            image = image.astype(np.uint8)

        return image

    def readInImage(self):
        image = mpimg.imread(self.chessboardImageName)
        if self.showOriginalImage:
            cv2.imshow(self.chessboardImageName, image)
            cv2.waitKey()
            cv2.destroyWindow(self.chessboardImageName)

        return image

    def handleRGBAndGrayscaleImages(self, image):
        self.imageIsGrayscale = True
        if self.verbose > 0:
            self.debugTabs.print('image shape: {}'.format(image.shape))
            self.debugTabs.print('image data type: {}'.format(image.dtype))

        if len(image.shape) == 3 and image.shape[2] == 3:
            if self.useColorImage:
                self.image = image
                self.imageIsGrayscale = False
            else:
                self.image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif len(image.shape) == 2 or image.shape[2] == 1:
            self.image = image
        else:
            if self.verbose > 0:
                self.debugTabs.print ('Unhandled image: {} unhandled shape: {}'.format(self.chessboardImageName, image.shape))

    def drawCorners(self, ret):
        # Convert image to color for display
        if self.imageIsGrayscale:
            displayImage = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
        else:
            displayImage = image

        # Draw image and display the corners
        cv2.drawChessboardCorners(displayImage, (self.nx, self.ny), self.corners, ret)

        if self.doLargerCircles:
            # Draw larger circles at chessboard corners for clarity
            for w in range(0,len(self.corners)):
                colorImg = cv2.circle(self.image, (self.corners[w][0], self.corners[w][1]), radius=24,color=(0,0,255), thickness=6)
            resultName = self.chessboardImageName

        cv2.imshow(self.chessboardImageName, displayImage)
        cv2.waitKey()
        cv2.destroyWindow(self.chessboardImageName)

    def loop(self, chessboardImageName):
        self.chessboardImageName = chessboardImageName
        image = self.readInImage()
        self.handleRGBAndGrayscaleImages(image)

        # Verify data is in 8-bit format (required by findChessboardCorners)
        self.image = self.convert2uint8(image)

        # Find the chessboard corners
        ret, self.corners = cv2.findChessboardCorners(image, (self.nx, self.ny), None)
        self.corners = np.squeeze(self.corners)
        # Print out coordinates of corners
        if self.verbose > 0:
            for c, corner in enumerate(self.corners):
                end = '   '
                row = c // self.nx
                column = c - row*self.nx
                if (c % self.nx) == 0:
                    self.debugTabs.print('row {}:'.format(c // self.nx), end=end)
                elif (c % self.nx) == (self.nx - 1):
                    end = '\n'
                self.debugTabs.print("({},{}): {}".format(row, column, corner), end=end)
        
        # If found, draw corners
        if ret == True:
            self.drawCorners(ret)
        else:
            self.debugTabs.print ('Unable to find chessboard corners for image {}'.format(self.chessboardImageName))

    def main(self):
        for chessboardImageName in self.chessboardImagesNames:
            self.loop(chessboardImageName)


"""
findChessboardCorners = FindChessboardCorners()
findChessboardCorners.main()
"""
