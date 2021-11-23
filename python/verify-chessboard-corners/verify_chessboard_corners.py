import os
import sys
currentPath = os.path.dirname(os.path.realpath(__file__))
corePath = os.path.join(currentPath, '../core')
corePath = os.path.realpath(corePath)
sys.path.append(corePath)

import find_chessboard_corners as fcc
import compatibility_chessboard as cc
import relax


compat = cc.CompatibilityChessboard(fcc.FindChessboardCorners())

print('run relax')

rl = relax.RelaxationLabeling(compat.compatibility, False)

print(' done with relax')


