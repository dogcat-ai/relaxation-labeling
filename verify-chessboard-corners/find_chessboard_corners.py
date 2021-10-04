import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import debug_tabs as dt

class FindChessboardCorners:
    def __init__(self):
        self.useColorImage = True
        self.showOriginalImage = True

        self.debugTabs = dt.DebugTabs()

        # Make a list of calibration images
        imageCase = 1
        if imageCase == 1:
            self.chessboardImagesNames = glob.glob('../../relaxation-labeling-supporting-files/verify-chessboard-corners/camera_cal/*.jpg')
            self.nx = 10
            self.ny = 7
        elif imageCase == 2:
            self.chessboardImagesNames = glob.glob('./supporting_files/stereo_rig/*.png')
            self.nx = 6
            self.ny = 5
        elif imageCase == 3:
            self.chessboardImagesNames = glob.glob('./supporting_files/calib_example1/*.tif')
            self.nx = 13
            self.ny = 12

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

    def main(self):

        # Select any index to grab an image from the list
        for chessboardImageName in self.chessboardImagesNames:
            # Read in the image
            chessboardImage = mpimg.imread(chessboardImageName)
            if self.showOriginalImage:
                cv2.imshow(chessboardImageName, chessboardImage)
                cv2.waitKey()
                cv2.destroyWindow(chessboardImageName)

            # Handle RGB and Grayscale images
            imageIsGrayscale = True
            self.debugTabs.print ('image shape {}'.format(chessboardImage.shape))
            self.debugTabs.print ('image data type {}'.format(chessboardImage.dtype))

            if len(chessboardImage.shape) == 3 and chessboardImage.shape[2] == 3:
                if self.useColorImage:
                    image = chessboardImage
                    imageIsGrayscale = False
                else:
                    image = cv2.cvtColor(chessboardImage, cv2.COLOR_RGB2GRAY)
            elif len(chessboardImage.shape) == 2 or chessboardImage.shape[2] == 1:
                image = chessboardImage
            else:
                self.debugTabs.print ('Unhandled image {} unhandled shape {}'.format(chessboardImageName, chessboardImage.shape))

            # Verify data is in 8-bit format (required by findChessboardCorners)
            image = self.convert2uint8(image)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(image, (self.nx, self.ny), None)
            corners = np.squeeze(corners)
            
            # If found, draw corners
            if ret == True:
                # Convert image to color for display
                if imageIsGrayscale:
                    displayImage = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                else:
                    displayImage = image

                # Draw image and display the corners
                cv2.drawChessboardCorners(displayImage, (self.nx, self.ny), corners, ret)

                if False:
                    # Draw larger circles at chessboard corners for clarity
                    for w in range(0,len(corners)):
                        colorImg = cv2.circle(image, (corners[w][0], corners[w][1]), radius=24,color=(0,0,255), thickness=6)
                    resultName = chessboardImageName

                cv2.imshow(chessboardImageName, displayImage)
                cv2.waitKey()
                cv2.destroyWindow(chessboardImageName)
            else:
                self.debugTabs.print ('Unable to find chessboard corners for image {}'.format(chessboardImageName))


findChessboardCorners = FindChessboardCorners()
findChessboardCorners.main()


