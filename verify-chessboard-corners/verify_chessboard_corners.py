# Here, we want to use the relaxation labeling algorithm to verify the chessboard corners found with OpenCV's findChessboardCorners function.

from find_chessboard_corners import FindChessboardCorners
from relax import RelaxationLabeling

class VerifyChessboardCorners(RelaxationLabeling):
    def __init__(self):
        self.findChessboardCorners = FindChessboardCorners()

