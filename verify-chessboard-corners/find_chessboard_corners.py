import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import debug_tabs as dt

def convert2uint8(image):
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


debugTabs = dt.DebugTabs()

USE_DEBUGTABS = True
USE_COLOR_IMAGE = True
SHOW_ORIGINAL_IMAGE = True


# Make a list of calibration images
Image_Case = 1
if Image_Case == 1:
    chessboard_images_names = glob.glob('./supporting_files/camera_cal/*.jpg')
    nx = 10
    ny = 7
elif Image_Case == 2:
    chessboard_images_names = glob.glob('./supporting_files/stereo_rig/*.png')
    nx = 6
    ny = 5
elif Image_Case == 3:
    chessboard_images_names = glob.glob('./supporting_files/calib_example1/*.tif')
    nx = 13
    ny = 12
# Select any index to grab an image from the list
for chessboard_image_name in chessboard_images_names:
    # Read in the image
    chessboard_image = mpimg.imread(chessboard_image_name)
    if SHOW_ORIGINAL_IMAGE:
        cv2.imshow(chessboard_image_name, chessboard_image)
        cv2.waitKey()
        cv2.destroyWindow(chessboard_image_name)

    # Handle RGB and Grayscale images
    image_is_grayscale = True
    debugTabs.print ('image shape {}'.format(chessboard_image.shape))
    debugTabs.print ('image data type {}'.format(chessboard_image.dtype))

    if len(chessboard_image.shape) == 3 and chessboard_image.shape[2] == 3:
        if USE_COLOR_IMAGE:
            image = chessboard_image
            image_is_grayscale = False
        else:
            image = cv2.cvtColor(chessboard_image, cv2.COLOR_RGB2GRAY)
    elif len(chessboard_image.shape) == 2 or chessboard_image.shape[2] == 1:
        image = chessboard_image
    else:
        debugTabs.print ('Unhandled image {} unhandled shape {}'.format(chessboard_image_name, chessboard_image.shape))

    # Verify data is in 8-bit format (required by findChessboardCorners)
    image = convert2uint8(image)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(image, (nx, ny), None)
    corners = np.squeeze(corners)
    
    # If found, draw corners
    if ret == True:
        # Convert image to color for display
        if image_is_grayscale:
            display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            display_image = image

        # Draw image and display the corners
        cv2.drawChessboardCorners(display_image, (nx, ny), corners, ret)

        if False:
            # Draw larger circles at chessboard corners for clarity
            for w in range(0,len(corners)):
                color_img = cv2.circle(image, (corners[w][0], corners[w][1]), radius=24,color=(0,0,255), thickness=6)
            result_name = chessboard_image_name

        cv2.imshow(chessboard_image_name, display_image)
        cv2.waitKey()
        cv2.destroyWindow(chessboard_image_name)
    else:
        debugTabs.print ('Unable to find chessboard corners for image {}'.format(chessboard_image_name))
