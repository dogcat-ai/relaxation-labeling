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

    def readImage(self):
        #imagePath = path + "../../relaxation-labeling-supporting-files/triangular-bond-w-1-offshoot.jpeg"
        #imagePath = os.path.expanduser("~/Code/relaxation-labeling-supporting-files/single_bonds.jpeg")
        imagePath = os.path.expanduser("../../relaxation-labeling-supporting-files/single_bonds.jpeg")
        self.image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        self.imageColor = cv2.imread(imagePath, cv2.IMREAD_COLOR)
        print("After initial imread, image's shape is:")
        print("\t{}".format(self.image.shape))
        #(thresh, self.image) = cv2.threshold(self.image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        #print("After Otsu thresholding, image's shape is:")
        #print("\t{}".format(self.image.shape))
        #cv2.imshow("molecule with single bonds", self.resize(self.image))
        #plt.title('Molecule with single bonds')
        #plt.imshow(self.image, cmap='gray')
        #plt.show()
        #cv2.waitKey()

    def doEdgeDetection(self):
        self.edgeImage = cv2.Canny(self.image,50,150,apertureSize = 3)

    def doHoughLinesP(self):
        #lines = cv2.HoughLinesP(self.edgeImage,1,np.pi/180,25,10,10)
        self.lines = cv2.HoughLinesP(self.edgeImage,rho = 1,theta = 1*np.pi/180,threshold = 35,minLineLength = 20,maxLineGap = 10)
        #self.lines = cv2.HoughLinesP(self.edgeImage,rho = 1,theta = 1*np.pi/180,threshold = 10,minLineLength = 5,maxLineGap = 10)
        if False:
            for l, line in enumerate(self.lines):
                for x1, y1, x2, y2 in line:
                    print("line #{}: {} {} {} {}".format(l, x1, y1, x2, y2))
                    self.imageColorCopy = self.imageColor.copy()
                    cv2.line(self.imageColorCopy,(x1,y1),(x2,y2),(0,0,255),2)
                    cv2.imwrite("line{:02d}.png".format(l), self.imageColorCopy)

            cv2.imshow("image with hough lines", self.imageColor)
            cv2.waitKey()
        self.lines = np.squeeze(self.lines)

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
        self.readImage()
        self.doEdgeDetection()
        self.doHoughLinesP()
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

    def getClusters(self, points, labels):
        clusters = []
        unique_labels = np.unique(labels)
        print 'labels',labels
        for label in unique_labels:
            indices = np.asarray(np.squeeze(np.asarray(np.where(labels == label))))
            print 'label',label,'indices',indices
            print 'Shape of indices',indices.shape


            cluster_points = np.asarray(points)[indices.astype('int')]
            cluster = []
            cluster.append(cluster_points)
            cluster.append(np.mean(cluster_points,axis=0))
            cluster.append(np.std(cluster_points,axis=0))
            clusters.append(cluster)
        return clusters

    def getMaxClusterStd(self, clusters):
        stds = np.stack(np.asarray(clusters)[:,2])
        norms = np.linalg.norm(stds, axis=1)
        return max(norms)

    def doClusterVertices(self, vertices, lines, distance_threshold, ClusteringTechnique='AGG', PlotClusters=True):
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
            while True:
                # Create Clusters
                hc = AgglomerativeClustering(n_clusters=number_clusters, affinity='euclidean',compute_full_tree=True)
                y_hc = hc.fit_predict(points)

                clusters = self.getClusters(points, y_hc)
                max_cluster_std = self.getMaxClusterStd(clusters)


                if PlotClusters:
                    plt.title('Clusters ('+str(number_clusters)+') with Stds -- Max = '+str(round(max_cluster_std,0)))
                    plt.imshow(self.image, cmap='gray', alpha=.25)
                    for cluster in clusters:
                        cluster_points = cluster[0]
                        cluster_mean = cluster[1]
                        cluster_std = cluster[2]
                        print 'cluster mean',cluster_mean,' shape',cluster_mean.shape
                        plt.plot(cluster_points[:,0], cluster_points[:,1],linewidth=0,marker='x')
                        plt.plot(cluster_mean[0], cluster_mean[1], marker='o', markersize=10)
                        plt.text(cluster_mean[0], cluster_mean[1], np.str(np.round(np.linalg.norm(cluster_std))))
                    plt.show()


                if max_cluster_std > .25*distance_threshold or number_clusters >= max_number_clusters:
                    number_clusters += 1
                else:
                    break

            if PlotClusters:
                plt.title('Vertices = '+str(number_clusters)+' Max Std='+str(round(max_cluster_std,0)))
                plt.imshow(self.image, cmap='gray', alpha=.25)
                for cluster in clusters:
                    cluster_points = cluster[0]
                    cluster_mean = cluster[1]
                    cluster_std = cluster[2]
                    print 'cluster mean',cluster_mean,' shape',cluster_mean.shape
                    plt.plot(cluster_mean[0], cluster_mean[1], marker='o', markersize=10, color='green')
                plt.show()

        return clusters, y_hc

    def end2end(self):
        self.readImage()
        plt.title('Molecule with single bonds')
        plt.imshow(self.image, cmap='gray')
        plt.show()

        self.doEdgeDetection()
        plt.title('Canny Edges of molecule')
        plt.imshow(self.edgeImage, cmap='gray')
        plt.show()

        self.doHoughLinesP()
        print 'Shape of HoughLinesP result',self.lines.shape
        plt.title('Hough Lines -- Count = '+str(self.lines.shape[0]))
        plt.imshow(self.edgeImage, cmap='gray')
        for line in self.lines:
            plt.plot([line[0],line[2]], [line[1],line[3]], marker='x', linestyle='solid', color='red')
        plt.show()

        typical_line_length = self.getTypicalLineLength(self.lines)
        print 'Typical Line Length = ', typical_line_length

        all_vertices = self.doComputeVertices(self.lines)
        print 'Shape of all vertices',all_vertices.shape

        vertices1 = self.doCleanVerticesByDistance(all_vertices, self.lines, typical_line_length)
        print 'Shape of vertices cleaned by distance of ',typical_line_length,' = ',vertices1.shape

        clean_distance = .25*typical_line_length
        vertices2 = self.doCleanVerticesByDistance(vertices1, self.lines, clean_distance)
        print 'Shape of vertices cleaned by distance of ',clean_distance,' = ',vertices2.shape
        plt.title('Vertices Cleaned -- Dist= '+ str(round(clean_distance,0))+'  Count='+str(vertices2.shape[0]))
        plt.imshow(self.image, cmap='gray')
        for vertex in vertices2:
            vertex_point = np.asarray(vertex[2])
            plt.plot(vertex_point[0], vertex_point[1], marker='x', color='red')
        plt.show()

        hc, y_hc = self.doClusterVertices(vertices2, self.lines, .2*typical_line_length)

        print 'hc =',hc
        print 'y_hc = ',y_hc



    def main(self):
        if True:
            self.end2end()
            print 'Number of line segments = ', self.lines.shape
        else:
            self.initLineSegmentObjectsAndLabels()
            super(MergeLineSegments, self).main()
            self.saveCompatibility("orientation_compatibility.csv",self.orientationCompatibility)
            self.saveCompatibility("proximity_compatibility.csv",self.proximityCompatibility)
            self.saveCompatibility("compatibility.csv",self.compatibility)


mergeLineSegments = MergeLineSegments()
mergeLineSegments.main()
