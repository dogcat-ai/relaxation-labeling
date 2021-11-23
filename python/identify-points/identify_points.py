import os
import sys
currentPath = os.path.dirname(os.path.realpath(__file__))
corePath = os.path.join(currentPath, '../core')
corePath = os.path.realpath(corePath)
sys.path.append(corePath)

import relax
import points3d_compatibility as pc
import datetime
import time

numLabels = 10
p3dc = pc.Points3DCompatibility(numLabels)

start_time = time.time()
rl = relax.RelaxationLabeling(p3dc.compatibility, False)
end_time = time.time()

exec_time = end_time - start_time
print('Execution Time (secs) = ',exec_time,'  Hz = ',1./exec_time)
