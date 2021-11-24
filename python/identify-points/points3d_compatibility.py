import numpy as np
import math

class Points3DCompatibility:
    def __init__(self, numLabels):
        self.DPSCALE = 2.0
        self.dim = 3
        self.numLabels = numLabels
        self.labels = np.random.rand(self.numLabels,self.dim)
        self.objects = np.copy(self.labels)

        # Modify Objects
        ROTATE_OBJECTS = True
        SCALE_OBJECTS = True
        TRANSLATE_OBJECTS = True
        ADD_NOISE_TO_OBJECTS = True
        REMOVE_OBJECTS = False
        ADD_OBJECTS = False
        FLIP_OBJECT_ORDER = True

        # Rotate, scale and translate objects wrt labels
        if ROTATE_OBJECTS:
            self.rotateObjects(np.deg2rad(35), np.deg2rad(55), np.deg2rad(70))
        if SCALE_OBJECTS:
            self.objects *= .2
        if TRANSLATE_OBJECTS:
            self.objects += [10, 20, 30]

        if FLIP_OBJECT_ORDER:
            # Just to be sure that relax is working properly, reverse the order
            # of the objects
            self.objects = np.flip(self.objects)
        if ADD_NOISE_TO_OBJECTS:
            noise = .01*np.random.rand(len(self.objects),self.dim)
            self.objects += noise

        if ADD_OBJECTS:
            self.objects = np.append(self.objects,[50, 60, 70]).reshape((-1,3))
            self.objects = np.append(self.objects, [0, 0, 0]).reshape((-1,3))
        if REMOVE_OBJECTS:
            self.objects = np.delete(self.objects, 0, axis=0).reshape((-1,3))
            self.objects = np.delete(self.objects, 0, axis=0).reshape((-1,3))

        self.numLabels = len(self.labels)
        self.numObjects = len(self.objects)

        print ('labels: ',self.labels)
        print ('objects: ', self.objects)
        self.calculate3DCompatibility()

    def rotateObjects(self, psi, theta, phi):
        mx = np.zeros((3,3))
        mx[0,0] =  1.0
        mx[1,1] =  math.cos(psi)
        mx[1,2] =  math.sin(psi)
        mx[2,2] =  math.cos(psi)
        mx[2,1] = -math.sin(psi)

        my = np.zeros((3,3))
        my[1,1] =  1.0
        my[0,0] =  math.cos(theta)
        my[0,2] = -math.sin(theta)
        my[2,2] =  math.cos(theta)
        my[2,0] =  math.sin(theta)

        mz = np.zeros((3,3))
        mz[2,2] =  1.0
        mz[0,0] =  math.cos(theta)
        mz[0,1] =  math.sin(theta)
        mz[1,1] =  math.cos(theta)
        mz[1,0] = -math.sin(theta)

        mxyz = np.matmul(mx, np.matmul(my, mz))

        for i in range(0,len(self.objects)):
            self.objects[i] = np.matmul(self.objects[i], mxyz)

    def calculate3DCompatibility(self):
        #TODO Add cross product results as part of compatibility
        # and verify that the final relax result is independent of
        # translation, scale and rotation
        self.compatibility = np.zeros((self.numObjects, self.numLabels, self.numObjects, self.numLabels, self.numObjects, self.numLabels))
        epsilon = 1e-8
        for i in range(self.numObjects):
            for j in range(self.numLabels):
                for k in range(self.numObjects):
                    for l in range(self.numLabels):
                        for m in range(self.numObjects):
                            for n in range(self.numLabels):
                                vobj21 = self.objects[k]-self.objects[i]
                                vobj23 = self.objects[m]-self.objects[i]
                                vlab21 = self.labels[l]-self.labels[j]
                                vlab23 = self.labels[n]-self.labels[j]
                                nvobj21 = np.linalg.norm(vobj21)
                                nvobj23 = np.linalg.norm(vobj23)
                                nvlab21 = np.linalg.norm(vlab21)
                                nvlab23 = np.linalg.norm(vlab23)
                                if nvobj21 > epsilon and nvobj23 > epsilon and nvlab21 > epsilon and nvlab23 > epsilon:
                                    vobj21 /= nvobj21
                                    vobj23 /= nvobj23
                                    vlab21 /= nvlab21
                                    vlab23 /= nvlab23
                                    self.compatibility[i,j,k,l,m,n] = 1.0/(1.0+self.DPSCALE*abs(np.dot(vobj21,vobj23) - np.dot(vlab21,vlab23)))


