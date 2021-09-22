import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

# prepare object points
#Enter the number of inside corners in x
nx = 10
nx = 5
#Enter the number of inside corners in y
ny = 7
ny = 6
# Make a list of calibration images
#chess_images = glob.glob('./camera_cal/*.jpg')
chess_images = glob.glob('./stereo_rig/*.png')
# Select any index to grab an image from the list
for i in range(len(chess_images)):
    # Read in the image
    chess_board_image = mpimg.imread(chess_images[i])
    print('shape of image',chess_board_image.shape)

    result_name = chess_images[i]
    cv2.imshow(result_name, chess_board_image)
    cv2.waitKey()

    if chess_board_image.shape[2] == 3:
        color_img = chess_board_image
        gray_img = cv2.cvtColor(chess_board_image, cv2.COLOR_RGB2GRAY)
    else:
        # Convert to color
        color_img = cv2.cvtColor(chess_board_image, cv2.COLOR_GRAY2RGB)
        gray_img = chess_board_image
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray_img, (nx, ny), None)
    print('corners shape')
    print(corners.shape)
    corners = np.squeeze(corners)
    print(corners.shape)
    # If found, draw corners
    if ret == True:
        # Draw and display the corners
        #cv2.drawChessboardCorners(chess_board_image, (nx, ny), corners, ret)
        cv2.drawChessboardCorners(color_img, (nx, ny), corners, ret)
        for w in range(0,len(corners)):
            color_img = cv2.circle(color_img, (corners[w][0], corners[w][1]), radius=24,color=(0,0,255), thickness=6)
        result_name = 'board'+str(i)+'.jpg'
        #cv2.imshow(result_name, chess_board_image)
        cv2.imshow(result_name, color_img)
        cv2.waitKey()
    else:
        print ('return is false')
