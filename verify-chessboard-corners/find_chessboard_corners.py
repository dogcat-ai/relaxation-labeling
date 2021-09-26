import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

USE_COLOR_IMAGE = True
SHOW_ORIGINAL_IMAGE = True

# Make a list of calibration images
chessboard_images_names = glob.glob('./supporting_files/camera_cal/*.jpg')
nx = 10
ny = 7
#chessboard_images_names = glob.glob('./supporting_files/stereo_rig/*.png')
#nx = 7
#ny = 6
# Select any index to grab an image from the list
for chessboard_image_name in chessboard_images_names:
    # Read in the image
    chessboard_image = mpimg.imread(chessboard_image_name)
    if SHOW_ORIGINAL_IMAGE:
        cv2.imshow(chessboard_image_name, chessboard_image)
        cv2.waitKey()

    # Handle RGB and Grayscale images
    image_is_grayscale = True
    print ('image shape',chessboard_image.shape)
    if len(chessboard_image.shape) == 3 and chessboard_image.shape[2] == 3:
        if USE_COLOR_IMAGE:
            image = chessboard_image
            image_is_grayscale = False
        else:
            image = cv2.cvtColor(chessboard_image, cv2.COLOR_RGB2GRAY)
    elif len(chessboard_image.shape) == 2 or chessboard_image.shape[2] == 1:
        image = chessboard_image
    else:
        print ('Unhandled image',chessboard_image_name,' unhandled shape',chessboard_image.shape)
        exit(1)

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
    else:
        print ('Unable to find chessboard corners for image',chessboard_image_name)
