import os
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import sklearn
from sklearn.cluster import KMeans
#user = "mannyglover"
#path = "/Users/" + user + "/Code/relaxation-labeling/core/"
path = "~/Code/relaxation-labeling/core/"
sys.path.append(os.path.dirname(os.path.expanduser(path)))
print(sys.path)
from relax import RelaxationLabeling

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ortho_group
from scipy.ndimage.morphology import binary_dilation, binary_erosion

# TODO Find vertices with number of lines == 1
# TODO Find vertex connections
# TODO Find Chemical characters at vertices
# TODO Create 'missing' chemical characters
# TODO Create Chemical Formula
# TODO Create International Chemical Identifier

class MergeLineSegments(RelaxationLabeling):
    def __init__(self):
        dim = 28
        maxNumPlots = 1
        noise = 0.0
        deleteLabel = -1
        objectOffset = 3*np.ones((dim))
        shuffleObjects = False
        objectScale = 2.0
        rotateObjects = False
        compatType = 2
        save = True
        super(MergeLineSegments, self).__init__(dim, maxNumPlots, noise, deleteLabel, objectOffset, shuffleObjects, objectScale, rotateObjects, compatType, save)

    def resize(self, img):
        factor = 3
        h, w = img.shape[:2]
        return cv2.resize(img, (w*factor, h*factor), interpolation=cv2.INTER_AREA)

    def readImage(self, imagePath):
        image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        imageColor = cv2.imread(imagePath, cv2.IMREAD_COLOR)
        print 'Initial Image shape',image.shape
        return image, imageColor

    def doEdgeDetection(self, image, MinThreshold=50, MaxThreshold=150, ApertureSize=3):
        edgeImage = cv2.Canny(image,MinThreshold,MaxThreshold,apertureSize = ApertureSize)
        return edgeImage

    def doHoughLinesP(self, image, case=1):
        all_lines = []
        for threshold in range(10,14):
            lines = cv2.HoughLinesP(image,rho = 1,theta = 1*np.pi/180,threshold = threshold,minLineLength = 10,maxLineGap = 5)
            lines = np.squeeze(lines)
            all_lines = np.reshape(np.append(all_lines, lines), (-1,lines.shape[1]))
        return all_lines

    def doHoughLines(self):
        lines = cv2.HoughLines(self.edgeImage,1,np.pi/180,50)
        for line in lines:
            for rho,theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
                cv2.line(self.imageColor,(x1,y1),(x2,y2),(0,0,255),2)
        cv2.imshow("image with hough lines", self.imageColor)
        cv2.waitKey()

    def initLineSegmentObjectsAndLabels(self):
        #imagePath = path + "../../relaxation-labeling-supporting-files/triangular-bond-w-1-offshoot.jpeg"
        #imagePath = os.path.expanduser("~/Code/relaxation-labeling-supporting-files/single_bonds.jpeg")
        imagePath = os.path.expanduser("../../relaxation-labeling-supporting-files/single_bonds.jpeg")
        #imagePath = os.path.expanduser("../../relaxation-labeling-supporting-files/000011a64c74.png")
        self.image, self.imageColor = self.readImage(imagePath)
        self.doEdgeDetection()
        self.lines = self.doHoughLinesP(self.edgeImage)
        self.objects = np.zeros(shape = [len(self.lines), 4])
        self.objectDistances = np.zeros(shape = [len(self.lines)])
        self.objectVectors = np.zeros(shape = [len(self.lines), 2])
        for l, line in enumerate(self.lines):
            self.objects[l] = line[0]
            self.objectVectors[l] = self.objects[1, 0:2] - self.objects[l, 2:]
            self.objectDistances[l] = np.linalg.norm(self.objectVectors[l])
        self.maxDistance = np.max(self.objectDistances)
        self.labels = self.objects
        self.labelDistances  = self.objectDistances
        self.labelVectors = self.objectVectors
        self.numObjects = len(self.objects)
        self.numLabels = len(self.labels)

    def calculateOrientationCompatibility(self, i, j, k, l):
        iVector = self.objectVectors[i]
        jVector = self.labelVectors[j]
        kVector = self.objectVectors[k]
        lVector = self.labelVectors[l]
        ijCompatibility = np.dot(iVector, jVector)/(self.objectDistances[i]*self.labelDistances[j])
        klCompatibility = np.dot(kVector, lVector)/(self.objectDistances[k]*self.labelDistances[l])
        self.orientationCompatibility[i, j, k, l] = 0.5*ijCompatibility + 0.5*klCompatibility

    def calculateOrientationCompatibilityVerbose(self, i, j, k, l):
        iObject = self.objects[i]
        jLabel = self.labels[j]
        kObject = self.objects[k]
        lLabel = self.labels[l]
        ijCompatibility = np.dot(iObject, jLabel)/(self.objectDistances[i]*self.labelDistances[j])
        ilCompatibility = np.dot(iObject, lLabel)/(self.objectDistances[i]*self.labelDistances[l])
        klCompatibility = np.dot(kObject, lLabel)/(self.objectDistances[k]*self.labelDistances[l])
        kjCompatibility = np.dot(kObject, jLabel)/(self.objectDistances[k]*self.labelDistances[j])
        # If object i is compatible with label j and object k is compatible with label j,
        # that is evidence that object i and object k belong to the same line segment.
        # So compatibility(i, j, k, j) should be high.
        # One way to express this mathematically is compatibility(i, j, k, j) = 0.5*ijCompatibility + 0.5*kjCompatibility
        # If object i is compatible with label j and object k is compatible with label l,
        # that is evidence that object i should have label j and object k should have label l, but not necessarily
        # that object i and object k go together.
        # If object i is compatible with label l and object k is compatible with label l,
        # that is evidence that object i and object k belong to the same line segment.
        # So compatibility(i, l, k, l) should be high.
        # One way to express this mathematically is compatibility(i, l, k, l) = 0.5*ilCompatibility + 0.5*klCompatibility
        self.compatibility[i, j, k, j] = 0.5*ijCompatibility + 0.5*kjCompatibility
        self.compatibility[i, l, k, l] = 0.5*ilCompatibility + 0.5*klCompatibility
        self.compatibility[i, j, k, l] = 0.5*ijCompatibility + 0.5*klCompatibility
        self.compatibility[i, l, k, j] = 0.5*ilCompatibility + 0.5*kjCompatibility
        self.compatibility[i, j, i, j] = ijCompatibility
        # If object i is compatible with label j and object k is compatible with label l,
        # that tells us that i could be j, and k could be l,
        # but what does it tell us about the compatibility of i being j AND k being l?
        # Does it tell us anything?
        # I suppose it at least tells us that this assignment is reasonable,
        # but I fear it will counter the merging we are trying to accomplish.
        # In my illustration, I say that compatibility(1, 1, 2, 2) should be low,
        # because we want 2 to belong to 1, or 1 to belong to 2,
        # but we don't want them to be independent.
        # But I don't think we want this compatibility to be negative.
        # Perhaps it should be zero?
        # We could express this mathematically with compatibility(i, j, k, l) = ijCompatibility - klCompatibility.
        # What does the above equation imply?  If ijCompatibility is high and klCompatibility is high,
        # the total compatibility is close to zero.
        # If ijCompatibility is high and klCompatibility is low,
        # the total compatibility is high.
        # But do we want this?
        # I think we do want this.
        # Because what it means is that i likes j, and k doesn't like l.
        # No, we don't want this.
        # If i likes j and k doesn't like l,
        # then the compatibility of (i,j,k,l) should be halfway between good and bad.
        # Mathematically, compatibility(i, j, k, l) = 0.5*ijCompatibility + 0.5*klCompatibility


    def calculateProximityCompatibility(self, i, j, k, l):
        # If the labels are identical, we want the objects to be close.
        # Otherwise, we want the objects to be close only if the labels are close.
        # We can express this mathematically like:
        #   if (j == l):
        #       compatibility = Constant - minDist(i, k)
        #   else:
        #       compatibility(i, j, k, l) = minDist(i, k) - minDist(j, l)
        # or:
        #       compatibility(i, j, k, l) = minDist(i, k)*minDist(j, l)?

        iObject = self.objects[i]
        kObject = self.objects[k]
        distance12 = np.linalg.norm(iObject[0:2] - kObject[2:])/self.maxDistance
        distance11 = np.linalg.norm(iObject[0:2] - kObject[0:2])/self.maxDistance
        distance21 = np.linalg.norm(iObject[2:] - kObject[0:2])/self.maxDistance
        distance22 = np.linalg.norm(iObject[2:] - kObject[2:])/self.maxDistance
        ikShortest = min(distance12, distance11, distance21, distance22)
        if j == l:
            self.proximityCompatibility[i, j, k, l] = 1 - ikShortest
        else:
            jLabel = self.labels[j]
            lLabel = self.labels[l]
            distance12 = np.linalg.norm(jLabel[0:2] - lLabel[2:])/self.maxDistance
            distance11 = np.linalg.norm(jLabel[0:2] - lLabel[0:2])/self.maxDistance
            distance21 = np.linalg.norm(jLabel[2:] - lLabel[0:2])/self.maxDistance
            distance22 = np.linalg.norm(jLabel[2:] - lLabel[2:])/self.maxDistance
            jlShortest = min(distance12, distance11, distance21, distance22)
            self.proximityCompatibility[i, j, k, l] = 1 - abs(ikShortest - jlShortest)

    def calculateCompatibility(self):
        self.orientationCompatibility = np.zeros((self.numObjects, self.numLabels, self.numObjects, self.numLabels))
        self.proximityCompatibility = np.zeros((self.numObjects, self.numLabels, self.numObjects, self.numLabels))
        self.compatibility = np.zeros((self.numObjects, self.numLabels, self.numObjects, self.numLabels))
        for i, iObject in enumerate(self.objects):
            for j, jLabel in enumerate(self.labels):
                for k, kObject in enumerate(self.objects):
                    for l, lLabel in enumerate(self.labels):
                        self.calculateOrientationCompatibility(i, j, k, l)
                        self.calculateProximityCompatibility(i, j, k, l)
                        self.compatibility[i, j, k, l] = 0.5*self.orientationCompatibility[i, j, k, l] + 0.5*self.orientationCompatibility[i, j, k, l]*self.proximityCompatibility[i, j, k, l]

    def computeIntersection(self, line1, line2):
        s = np.vstack([line1[:2], line1[2:], line2[:2], line2[2:]])
        h = np.hstack((s, np.ones((4,1))))
        l1 = np.cross(h[0], h[1])
        l2 = np.cross(h[2], h[3])
        x, y, z = np.cross(l1, l2)
        if z==0:
            return (float('inf'), float('inf'))
        else:
            return (x/z, y/z)

    def doComputeVertices(self, lines):
        vertices = []
        for i in range(0,lines.shape[0]):
            for j in range(i+1,lines.shape[0]):
                vertices.append([i, j, self.computeIntersection(lines[i], lines[j])])
        return np.asarray(vertices)

    def getLineLength(self, line):
        length = np.linalg.norm([line[2]-line[0], line[3]-line[1]])
        return length

    def getTypicalLineLength(self, lines):
        lineLengths = []
        for line in lines:
            lineLengths.append(self.getLineLength(line))
        return np.median(lineLengths)

    def distanceFromVertexToLine(self, vertex, line):
        d2Point1 = np.linalg.norm(vertex[2]-line[:2])
        d2Point2 = np.linalg.norm(vertex[2]-line[2:])
        return min(d2Point1, d2Point2)

    def doCleanVerticesByDistance(self, vertices, lines, line_length_threshold):
        cleaned_vertices = []
        for vertex in vertices:
            d1 = self.distanceFromVertexToLine(vertex, lines[vertex[0]])
            d2 = self.distanceFromVertexToLine(vertex, lines[vertex[1]])
            d  = max(d1, d2)
            if d < line_length_threshold:
                cleaned_vertices.append(vertex)
        return np.asarray(cleaned_vertices)


    def doGetMaximumOutlierPerGroup(points, labels):
        cluster_number = 0
        while True:
            cluster = self.getClusterNumber(points, labels, cluster_number)
            if cluster.shape[0] == 0:
                break
            distance = self.getClusterMaximumOutlierDistance(cluster)
            outliers.append(distance)
        return np.asarray(outliers)

    def getPointsFromVertices(self, vertices):
        points = []
        for vertex in vertices:
            points.append(np.asarray(vertex[2]))
        return np.asarray(points)

    def getClusters(self, vertices, labels):
        clusters = []
        unique_labels = np.unique(labels)
        print 'labels',labels
        for label in unique_labels:
            #indices = np.asarray(np.squeeze(np.asarray(np.where(labels == label))))
            indices = np.squeeze(np.asarray(np.where(labels == label)))

            cluster_vertices = np.asarray(vertices)[indices.astype('int')]
            print 'indices',indices,'  size',np.size(indices),'  ndim',np.ndim(indices),'  type',type(indices)
            if np.ndim(indices) == 0:
                indices = np.array([int(indices)])
                print ' mod indices',indices,'  isscalar',np.isscalar(indices),'  ndim',np.ndim(indices)
                cluster_vertices = np.asarray(vertices)[indices.astype('int')]
                print 'Shape of cluster_vertices',cluster_vertices.shape
            print 'Shape of vertices',vertices.shape
            print 'Shape of cluster_vertices',cluster_vertices.shape
            cluster = []
            cluster.append(cluster_vertices)
            points = self.getPointsFromVertices(cluster_vertices)
            cluster.append(np.mean(points,axis=0))
            cluster.append(np.std(points,axis=0))
            clusters.append(cluster)
            
        return np.asarray(clusters)

    def getMaxClusterStd(self, clusters):
        stds = np.stack(np.asarray(clusters)[:,2])
        norms = np.linalg.norm(stds, axis=1)
        return max(norms)

    def doClusterVertices(self, vertices, lines, distance_threshold, ClusteringTechnique='AGG', PlotClusters=False):
        import scipy.cluster.hierarchy as sch
        from sklearn.cluster import AgglomerativeClustering

        points = []
        for vertex in vertices:
            points.append(np.asarray(vertex[2]))

        if ClusteringTechnique == 'AGG':

            # Create dendrogram
            dendrogram = sch.dendrogram(sch.linkage(points, method='ward'))
            plt.title('AGG Result')
            plt.show()

            number_clusters = 5
            max_number_clusters = 50
            remaining_points = points
            remaining_vertices = vertices
            final_clusters = []
            while True:
                # Create Clusters
                hc = AgglomerativeClustering(n_clusters=number_clusters, affinity='euclidean',compute_full_tree=True)
                y_hc = hc.fit_predict(remaining_points)

                clusters = self.getClusters(remaining_vertices, y_hc)
                max_cluster_std = self.getMaxClusterStd(clusters)

                if False:
                    good_clusters, remaining_vertices = self.getGoodClusters(clusters, vertices)
                    remaining_points = []
                    for vertex in remaining_vertices:
                        remaining_points.append(np.asarray(vertex[2]))

                    if np.size(good_clusters) != 0:
                        final_clusters.append(good_clusters)
                        number_clusters -= good_clusters.shape[0]

                if PlotClusters:
                    plt.title('Clusters ('+str(number_clusters)+') with Stds -- Max = '+str(round(max_cluster_std,0))+'  Threshold='+str(round(distance_threshold)))
                    plt.imshow(self.image, cmap='gray', alpha=.25)
                    for cluster in clusters:
                        print 'shape of cluster',cluster.shape
                        print 'first element of cluster',cluster[0]
                        print 'shape of first element of cluster',cluster[0].shape
                        cluster_points = self.getPointsFromVertices(cluster[0])
                        cluster_mean = cluster[1]
                        cluster_std = cluster[2]
                        print 'cluster mean',cluster_mean,' shape',cluster_mean.shape
                        plt.plot(cluster_points[:,0], cluster_points[:,1],linewidth=0,marker='x')
                        plt.plot(cluster_mean[0], cluster_mean[1], marker='o', markersize=10)
                        plt.text(cluster_mean[0], cluster_mean[1], np.str(np.round(np.linalg.norm(cluster_std))))
                    plt.show()


                if max_cluster_std > distance_threshold or number_clusters >= max_number_clusters:
                    number_clusters += 1
                else:
                    break

            if PlotClusters:
                plt.title('Vertices = '+str(number_clusters)+' Max Std='+str(round(max_cluster_std,0)))
                plt.imshow(self.image, cmap='gray', alpha=.25)
                for cluster in clusters:
                    cluster_points = cluster[0][:,2]
                    cluster_mean = cluster[1]
                    cluster_std = cluster[2]
                    print 'cluster mean',cluster_mean,' shape',cluster_mean.shape
                    plt.plot(cluster_mean[0], cluster_mean[1], marker='o', markersize=10, color='green')
                plt.show()

        return clusters, y_hc

    def lengthOfLine(self, line):
        return np.linalg.norm(line[:2] - line[2:])

    def getCollapsedDistanceLineToPoints(self, line, Pi, Pj):
        ds = []
        line_length = self.lengthOfLine(line)

        # Note. The below sequence of length calculations cannot be modified 
        # independently of the calculation below for length2line2
        ds.append(np.linalg.norm(Pi-line[:2]))
        ds.append(np.linalg.norm(Pi-line[2:]))
        ds.append(np.linalg.norm(Pj-line[:2]))
        ds.append(np.linalg.norm(Pj-line[2:]))
        ds = np.asarray(ds)
        index1 = np.argmin(ds)
        length2line1 = ds[index1]
        length2line2 = ds[3-index1]

        d = line_length + length2line1 + length2line2
        excess = length2line1 + length2line2
        return d, excess

    def getLineVectorDotProduct(self, line, vec):
        linevec = np.asarray(line[:2]-line[2:])
        print 'linevec',linevec,'  shape',linevec.shape
        print 'vec',vec,'  shape',vec.shape
        dp = np.dot(vec,linevec)/(np.linalg.norm(vec)*np.linalg.norm(linevec))
        return abs(dp)


    def doConnectClusters(self, clusters, lines, typical_line_length, DistanceThreshold=.45, DotpThreshold=.025, PlotDebug=False):

        if False:
            for i in range(0,clusters.shape[0]):
                vertices = clusters[i][0]
                mean = clusters[i][1]
                std = clusters[i][2]
                plt.title('Clusters #'+str(i)+' vertices'+'  std='+str(std))
                plt.imshow(self.image, cmap='gray', alpha=.25)
                plt.plot(mean[0], mean[1], marker='+', markersize=15, color='blue')
                for vertex in vertices:
                    point = vertex[2]
                    plt.plot(point[0], point[1], marker='x', markersize=10, color='red')
                plt.show()


        if True:
            connections = []
            print 'doConnectClusters...........'
            if PlotDebug:
                plt.title('Debug connections')
                plt.imshow(self.image, cmap='gray', alpha=.25)
            for i in range(0,clusters.shape[0]):
                Pi = clusters[i][1]
                vertices_i = clusters[i][0]
                for j in range(i+1,clusters.shape[0]):
                    Pj = clusters[j][1]
                    if PlotDebug:
                        plt.plot(Pi[0], Pi[1], marker='x', markersize=10, color='red')
                        plt.plot(Pj[0], Pj[1], marker='x', markersize=10, color='green')
                    vertices_j = clusters[j][0]
                    Vecij = Pi - Pj
                    distij = np.linalg.norm(Vecij)
                    
                    Pij_connected = False
                    print 'DistanceThreshold',DistanceThreshold,'DotpThreshold',DotpThreshold,'excess threshold',.5*typical_line_length
                    for vertex in vertices_i:
                        Pij_connected = False
                        for index in vertex[:2]:
                            Pij_connected = False
                            line = self.lines[index]
                            dp = self.getLineVectorDotProduct(line, Vecij)
                            d, excess = self.getCollapsedDistanceLineToPoints(line, Pi, Pj)
                            fom = abs(d-distij)/distij
                            print 'i',i,'j',j,'fom',fom,'abs(dp-1)',abs(dp-1),'excess',excess
                            if fom < DistanceThreshold and abs(dp-1) < DotpThreshold and excess < .5*typical_line_length:
                                if PlotDebug:
                                    plt.title('i='+str(i)+' j='+str(j)+'  d='+str(round(d,0)) + '  dij='+str(round(distij))+'  dp='+str(round(dp,3)))
                                    plt.imshow(self.image, cmap='gray', alpha=.25)
                                    plt.plot([line[0],line[2]], [line[1],line[3]], linewidth=1, color='blue')
                                    plt.text(.5*(line[0]+line[2]), .5*(line[1]+line[3]), str(round(fom,2)))
                                    plt.plot([Pi[0],Pj[0]], [Pi[1],Pj[1]], linewidth=1, color='red')
                                    plt.show()
                                Pij_connected = True
                                connections.append([i,j])
                                break
                            if Pij_connected:
                                break
                        if Pij_connected:
                            break

                    if Pij_connected == False:
                        for vertex in vertices_j:
                            Pij_connected = False
                            for index in vertex[:2]:
                                Pij_connected = False
                                line = self.lines[index]
                                dp = self.getLineVectorDotProduct(line, Vecij)
                                d, excess = self.getCollapsedDistanceLineToPoints(line, Pi, Pj)
                                fom = abs(d-distij)/distij
                                if fom < DistanceThreshold and abs(dp-1)<DotpThreshold and excess < .5*typical_line_length:
                                    if PlotDebug:
                                        plt.title('**i='+str(i)+' j='+str(j)+'  d='+str(round(d,0)) + '  dij='+str(round(distij))+'  dp='+str(round(dp,3)))
                                        plt.imshow(self.image, cmap='gray', alpha=.25)
                                        plt.plot([line[0],line[2]], [line[1],line[3]], linewidth=1, color='blue')
                                        plt.text(.5*(line[0]+line[2]), .5*(line[1]+line[3]), str(round(fom,2)))
                                        plt.plot([Pi[0],Pj[0]], [Pi[1],Pj[1]], linewidth=1, color='red')
                                        plt.show()
                                    Pij_connected = True
                                    connections.append([i,j])
                                    break
                                if Pij_connected:
                                    break
                            if Pij_connected:
                                break

        return np.asarray(connections)

    def linesAreEqual(self, line1, line2, epsilon=.01):
        dx1 = abs(line1[0] - line2[0])
        dy1 = abs(line1[1] - line2[1])
        dx2 = abs(line1[2] - line2[2])
        dy2 = abs(line1[3] - line2[3])
        d = dx1 + dy1 + dx2 + dy2
        return d < epsilon

    def linesAreClose(self, line1, line2, epsilon=1.01):
        dx1 = abs(line1[0] - line2[0])
        dy1 = abs(line1[1] - line2[1])
        dx2 = abs(line1[2] - line2[2])
        dy2 = abs(line1[3] - line2[3])
        return dx1 < epsilon and dy1 < epsilon and dx2 < epsilon and dy2 < epsilon

    def mergeHoughLines(self, lines):
        print 'mergeHoughLines'
        print 'Initial shape of lines',lines.shape
        #lines_sorted = lines[np.argsort(lines[:,0])]
        lines_sorted = lines[np.lexsort((lines[:,3], lines[:,2], lines[:,1], lines[:,0]))]

        # Remove identical lines
        clean_lines = []
        last_line = lines_sorted[0]
        clean_lines = np.append(clean_lines, last_line)
        for i in range(1,lines_sorted.shape[0]):
            next_line = lines_sorted[i]
            if self.linesAreEqual(last_line, next_line):
                continue
            else:
                last_line = next_line
                clean_lines = np.append(clean_lines, last_line)
        clean_lines = np.reshape(clean_lines, (-1, lines_sorted.shape[1]))
        print 'shape of clean lines',clean_lines.shape

        # Merge close lines
        merged_lines = []
        last_line = clean_lines[0]
        i = 1
        while i < clean_lines.shape[0]:
            next_line = clean_lines[i]
            if self.linesAreClose(last_line, next_line, epsilon=1.01):
                merged_line = .5*(last_line + next_line)
                merged_lines = np.append(merged_lines, merged_line)

                i += 1
                if i < clean_lines.shape[0]:
                    last_line = clean_lines[i]
                    last_line_merged = False
                else:
                    last_line_merged = True

                i += 1

            else:
                merged_lines = np.append(merged_lines, last_line)
                last_line = next_line
                last_line_merged = False
                i += 1

        if not last_line_merged:
                merged_lines = np.append(merged_lines, last_line)
        merged_lines = np.reshape(merged_lines, (-1,clean_lines.shape[1]))
        print 'shape of merged lines',merged_lines.shape

        return merged_lines


    def end2end(self):
        import pathlib

        NumberDilationErodeIterations = 1

        #imagePath = os.path.expanduser("../../relaxation-labeling-supporting-files/single_bonds.jpeg")
        imageDir = "../../relaxation-labeling-supporting-files"
        imageNames = [f for f in os.listdir(imageDir) if pathlib.Path(f).suffix == '.png']
        for imageName in imageNames:
            imagePath = os.path.join(imageDir, imageName)
            self.image, self.imageColor = self.readImage(imagePath)
            plt.title(imageName + ' of shape='+np.str(self.image.shape))
            plt.imshow(self.image, cmap='gray')
            plt.show()

            # If image is black lines on white background, 
            # then image needs to be reversed before dilation.
            imageReversed = np.asarray(255 - self.image).astype('uint8')
            imageDilated = binary_dilation(imageReversed, iterations=NumberDilationErodeIterations)
            #imageDilated = np.asarray(255 - imageDilated).astype('uint8')
            plt.title('Dilated Image')
            plt.imshow(imageDilated, cmap='gray')
            plt.show()

            imageEroded = binary_erosion(imageDilated, iterations=NumberDilationErodeIterations)
            self.houghImage = imageEroded
            self.houghImage = np.asarray(self.houghImage).astype('uint8')
            self.houghImage[self.houghImage != 0] = 255
            plt.title('Eroded Image (image to process)')
            plt.imshow(self.houghImage, cmap='gray')
            plt.show()

            self.lines = self.doHoughLinesP(self.houghImage)

            plt.title('Hough Lines -- Count = '+str(self.lines.shape[0]))
            plt.imshow(self.houghImage, cmap='gray')
            for line in self.lines:
                plt.plot([line[0],line[2]], [line[1],line[3]], marker='x', linestyle='solid', color='red')
            plt.show()

            self.lines = self.mergeHoughLines(self.lines)

            plt.title('Merged Hough Lines -- Count = '+str(self.lines.shape[0]))
            plt.imshow(self.houghImage, cmap='gray')
            for line in self.lines:
                plt.plot([line[0],line[2]], [line[1],line[3]], marker='x', linestyle='solid', color='red')
            plt.show()


            all_vertices = self.doComputeVertices(self.lines)

            typical_line_length = self.getTypicalLineLength(self.lines)
            #clean_distance = .25*typical_line_length
            clean_distance = .35*typical_line_length
            self.vertices = self.doCleanVerticesByDistance(all_vertices, self.lines, clean_distance)
            plt.title('Intersection of hough lines (Vertices) \n restricted by intersection to closest line endpoint <'+ str(round(clean_distance,2))+' pixels')
            plt.imshow(self.houghImage, cmap='gray')
            for vertex in self.vertices:
                vertex_point = np.asarray(vertex[2])
                plt.plot(vertex_point[0], vertex_point[1], marker='x', color='red')
            plt.show()

            #clusters, labels = self.doClusterVertices(self.vertices, self.lines, .114*typical_line_length)
            clusters, labels = self.doClusterVertices(self.vertices, self.lines, .15*typical_line_length)

            plt.title('Means of clustered vertices -- Count='+str(clusters.shape[0]))
            plt.imshow(self.houghImage, cmap='gray')
            for i,cluster in enumerate(clusters):
                location = cluster[1]
                plt.plot(location[0], location[1], marker='+', markersize=10, color='red')
                plt.text(location[0], location[1], '#'+str(i))
            plt.show()

            connections = self.doConnectClusters(clusters, self.lines, typical_line_length)
            print 'connections',connections

            plt.title('Clusters Connected')
            plt.imshow(self.image, cmap='gray', alpha=.25)
            for i,cluster in enumerate(clusters):
                location = cluster[1]
                plt.plot(location[0], location[1], marker='+', markersize=10, color='red')
                plt.text(location[0], location[1], '#'+str(i))
            for connection in connections:
                cluster1_mean = clusters[connection[0]][1]
                cluster2_mean = clusters[connection[1]][1]
                plt.plot([cluster1_mean[0],cluster2_mean[0]], [cluster1_mean[1],cluster2_mean[1]], linewidth=1, color='green')
                #plt.text(.5*(cluster1_mean[0]+cluster2_mean[0]), .5*(cluster1_mean[1]+cluster2_mean[1]), str(connection[0])+' to '+str(connection[1]))
            plt.show()



    def main(self):
        if True:
            self.end2end()
            print 'Number of line segments = ', self.lines.shape
        else:
            self.initLineSegmentObjectsAndLabels()
            super(MergeLineSegments, self).main()
            self.saveCompatibilityForPlotting("orientation_compatibility_plt.csv",self.orientationCompatibility)
            self.saveCompatibilityForPlotting("proximity_compatibility_plt.csv",self.proximityCompatibility)
            self.saveCompatibilityForPlotting("compatibility_plt.csv",self.compatibility)
            self.saveCompatibility("orientation_compatibility.csv",self.orientationCompatibility)
            self.saveCompatibility("proximity_compatibility.csv",self.proximityCompatibility)
            self.saveCompatibility("compatibility.csv",self.compatibility)


mergeLineSegments = MergeLineSegments()
mergeLineSegments.main()
