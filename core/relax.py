import numpy as np
import random
from itertools import permutations

class RelaxationLabeling(object):

    # n-dimensional cartesian points -> n-dimensional cartesian points
    # n-dimensional cartesian points -> m-dimensional (projections) cartesian points
    # n-dimensional lists -> n-dimensional lists
    # n-dimensional lists -> m-dimensional lists
    # labeling of chemical molecules
    # compat speed up
    #   multiplication of vector (and or matrices) not just elements one at a time
    # generalization of n-dim compat

    def __init__(self):
        pass

    def initListObjectsAndLabels(self, Dim, NumObjects, NumLabels, Noise, DeleteLabel, ObjectOffset, ObjectScale, ShuffleObjects):
        labels = np.random.rand(NumLabels,Dim)

        objects = np.copy(labels)
        if ShuffleObjects:
            np.random.shuffle(objects)

        if Noise > 0.0:
            noise = Noise*np.random.rand(objects.shape)
            objects += Noise

        if DeleteLabel >= 0:
            labels = np.delete(labels, DeleteLabel, axis=0)
            #objects = np.append(objects, [[50.14156, 70.7134]], axis=0)

        if ObjectOffset[0] > 0.0:
            objects += ObjectOffset

        if ObjectScale != 1.0:
            objects *= ObjectScale

        return objects, labels

    def initListCompat(self, objects, labels):

        NumObjects = objects.shape[0]
        NumLabels = labels.shape[0]
        r = np.zeros((NumObjects, NumLabels, NumObjects, NumLabels))

        epsilon = 1e-8

        UseDistance = False
        UseAngle = True
        AngleTechnique = 3
        for i in range(0,NumObjects):
            for j in range(0,NumLabels):
                for k in range(0,NumObjects):
                    for l in range(0,NumLabels):
                        if UseDistance:
                            r[i,j,k,l] = 1.0/(1.0+np.abs(np.linalg.norm(objects[i]-objects[k]) - np.linalg.norm(labels[j]-labels[l])))
                        if UseAngle:
                            vobj = objects[i]-objects[k]
                            vlab = labels[j]-labels[l]
                            if np.linalg.norm(vobj) < epsilon or np.linalg.norm(vlab) < epsilon:
                                r[i,j,k,l] = 0.0
                            else:
                                vobj = vobj/np.linalg.norm(vobj)
                                vlab = vlab/np.linalg.norm(vlab)
                                dot = np.dot(vobj, vlab)
                                if AngleTechnique == 1:
                                    r[i,j,k,l] = dot
                                elif AngleTechnique == 2:
                                    if dot < 0:
                                        r[i,j,k,l] = 0.0
                                    else:
                                        r[i,j,k,l] = dot
                                elif AngleTechnique == 3:
                                    if dot < 0:
                                        r[i,j,k,l] = -dot*dot
                                    else:
                                        r[i,j,k,l] = dot*dot

        p = np.zeros((NumObjects, NumLabels))
        for i in range(0,NumObjects):
            for j in range(0,NumLabels):
                p[i,j] = 1.0/NumLabels

        return r, p

    # Points
    #   n-dimensional real-valued coordinates
    #   nxm real-valued matrices
    #   list of n elements, each element can be 'anything', just so we can define some sort of metric, dot-product, ....

    # TODO
    #   Implement options for each of the types of points
    #   Extend compat from i,j,k,l to any number
    #   How was Manny handling the Cardlytics problem
    #       - Which type of point was he using
    #   Find some problems that fit this type of solution
    #   Use the current problem type, i.e, rearranging of points, to test the above point type classes



    def init2DCompat(self, objects, labels):

        NumObjects = objects.shape[0]
        NumLabels = labels.shape[0]
        r = np.zeros((NumObjects, NumLabels, NumObjects, NumLabels))

        epsilon = 1e-8

        UseDistance = False
        UseAngle = True
        AngleTechnique = 3
        for i in range(0,NumObjects):
            for j in range(0,NumLabels):
                for k in range(0,NumObjects):
                    for l in range(0,NumLabels):
                        if UseDistance:
                            r[i,j,k,l] = 1.0/(1.0+np.abs(np.linalg.norm(objects[i]-objects[k]) - np.linalg.norm(labels[j]-labels[l])))
                        if UseAngle:
                            vobj = objects[i]-objects[k]
                            vlab = labels[j]-labels[l]
                            if np.linalg.norm(vobj) < epsilon or np.linalg.norm(vlab) < epsilon:
                                r[i,j,k,l] = 0.0
                            else:
                                vobj = vobj/np.linalg.norm(vobj)
                                vlab = vlab/np.linalg.norm(vlab)
                                dot = np.dot(vobj, vlab)
                                if AngleTechnique == 1:
                                    r[i,j,k,l] = dot
                                elif AngleTechnique == 2:
                                    if dot < 0:
                                        r[i,j,k,l] = 0.0
                                    else:
                                        r[i,j,k,l] = dot
                                elif AngleTechnique == 3:
                                    if dot < 0:
                                        r[i,j,k,l] = -dot*dot
                                    else:
                                        r[i,j,k,l] = dot*dot

        p = np.zeros((NumObjects, NumLabels))
        for i in range(0,NumObjects):
            for j in range(0,NumLabels):
                p[i,j] = 1.0/NumLabels

        return r, p


    def updateSupport(self, CompatType, NumObjects, NumLabels, s, p, r):
        if CompatType == 2:
            for i in range(0,NumObjects):
                for l in range(0, NumLabels):
                    s[i,l] = 0.0
                    for j in range(0,NumObjects):
                        for lp in range(0,NumLabels):
                            s[i,l] += p[j,lp]*r[i,l,j,lp]
                normalizeSupport(i, s)

        if CompatType == 3:
            for i1 in range(0,NumObjects):
                for l1 in range(0, NumLabels):
                    for i2 in range(0,NumObjects):
                        for l2 in range(0,NumLabels):
                            for i3 in range(0,NumObjects):
                                for l3 in range(0,NumLabels):
                                    s[i1,l1] += p[i2,l2]*p[i3,l3]*r[i1,l1,i2,l2,i3,l3]

                normalizeSupport(i1, s)


    def normalizeSupport(self, i, s):
        minimumSupport = np.amin(s[i,:])
        maximumSupport = np.amax(s[i,:])
        maximumSupport -= minimumSupport
        s[i, :] = (s[i, :] - minimumSupport)/maximumSupport


    def updateProbability(self, NumObjects, NumLabels, s, p, r, SupportFactor):
        Technique = 2
        if Technique == 1:
            for i in range(0,NumObjects):
                for l in range(0,NumLabels):
                    p[i,l] += s[i,l]*SupportFactor
                normalizeProbability(i, p)
        if Technique == 2:
            for i in range(0,NumObjects):
                den = 0.0
                for lp in range(0,NumLabels):
                    den += p[i,lp]*(1.0+s[i,lp])
                for l in range(0,NumLabels):
                    p[i,l] = p[i,l]*(1.0+s[i,l])/den
                normalizeProbability(i, p)


    def normalizeProbability(self, i, p):
        Technique = 2
        if Technique == 1 or Technique == 2:
            minProbability = np.amin(p[i, :])
            p[i, :] -= minProbability
            if Technique == 2:
                sumProbability = np.sum(p[i, :])
                p[i, :] /= sumProbability
                #sumProbability = np.sum(p[i, :])
                #if abs(sumProbability - 1.0) > 1e-6:
                #    print(' probabiility error sumProbability = ',sumProbability)
                #    exit()

