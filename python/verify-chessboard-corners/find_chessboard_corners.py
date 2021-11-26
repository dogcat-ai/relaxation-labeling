import cv2
import glob
import copy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import debug_tabs as dt

class FindChessboardCorners:
    def __init__(self, useMatPlotLib=True, showImageOrCorners=True):
        self.debugTabs = dt.DebugTabs()
        self.verbose = 1
        self.useMatPlotLib = useMatPlotLib
        self.showImageOrCorners = showImageOrCorners
        self.useColorImage = True
        self.doLargerCircles = False
        self.corners = []

        # Make a list of calibration images
        imageCase = 1
        if imageCase == 1:
            self.chessboardImagesNames = glob.glob('./images/camera_cal/*.jpg')
            self.numColumns = 10
            self.numRows = 7
        elif imageCase == 2:
            self.chessboardImagesNames = glob.glob('./images/stereo_rig/*.png')
            self.numColumns = 6
            self.numRows = 5
        elif imageCase == 3:
            self.chessboardImagesNames = glob.glob('./images/calib_example1/*.tif')
            self.numColumns = 13
            self.numRows = 12

        if self.verbose > 0:
            self.debugTabs.print("chessboard images names: {}".format(self.chessboardImagesNames))
            self.debugTabs.print("num columns: {}".format(self.numColumns))
            self.debugTabs.print("num rows: {}".format(self.numRows))

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
        if self.showImageOrCorners:
            if self.useMatPlotLib:
                plt.figure(self.chessboardImageName)
                plt.title('Original Image')
                plt.imshow(image, cmap='gray')
                plt.show()
            else:
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

        if self.useMatPlotLib:
            plt.figure(self.chessboardImageName)
            plt.title('Chessboard Corners')
            plt.imshow(displayImage)
            x = []
            y = []
            for w in range(0,len(self.corners)):
                x.append(self.corners[w][0])
                y.append(self.corners[w][1])
            plt.plot(x, y, marker='x', color='red', linestyle='dashed', linewidth=1)
            plt.show()
        else:
            # Draw image and display the corners
            cv2.drawChessboardCorners(displayImage, (self.numColumns, self.numRows), self.corners, ret)

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
        ret, self.corners = cv2.findChessboardCorners(image, (self.numColumns, self.numRows), None)
        self.corners = np.squeeze(self.corners)
        # Print out coordinates of corners
        if self.verbose > 0:
            for c, corner in enumerate(self.corners):
                end = '   '
                row = c // self.numColumns
                column = c - row*self.numColumns
                if (c % self.numColumns) == 0:
                    self.debugTabs.print('row {}:'.format(c // self.numColumns), end=end)
                elif (c % self.numColumns) == (self.numColumns - 1):
                    end = '\n'
                self.debugTabs.print("({},{}): {}".format(row, column, corner), end=end)
        
        # If found, draw corners
        if ret == True:
            if self.showImageOrCorners:
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
