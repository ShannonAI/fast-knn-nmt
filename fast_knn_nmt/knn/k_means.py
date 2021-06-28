import numpy as np
#from queue import PriorityQueue
import heapq

class KMeans:

    def __init__(self, n_clusters, max_iter=20, tol=0.2):
        """
        n_clusters : int
        The number of clusters to form as well as the number of
        centroids to generate.

        max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a
        single run.

        tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
        """

        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def _inital_centers(self, X):
        n_features = X.shape[1]
        centers = np.zeros((self.n_clusters, n_features))
        for i in range(n_features):
            min_ = min(X[:, i])
            max_ = max(X[:, i])
            span_ = float(max_ - min_)
            centers[:, i] = min_ + span_ * np.random.rand(self.n_clusters, 1).reshape(self.n_clusters, )
        return centers

    def _vector_distance(self, x, y):
        return np.sqrt(sum(np.power(x-y, 2)))

    def fit(self, X, centers_max_num):
        """
        X : array of shape (n_samples, n_features)
            Training instances to cluster.
        
        centers_max_num : max number of point in one cluster
        """

        n_samples = X.shape[0]
        clustser_dis = np.zeros(n_samples)
        clustser_label = np.zeros(n_samples, dtype=np.int32)
        centers = self._inital_centers(X)
        cluster_change = True
        now_iter = 0
        #max_cluster = np.zeros(self.n_clusters)

        if (self.n_clusters == 1):
            self.n_iter_ = 0
            centers[0, :] = X[0, :]
            self.cluster_centers = centers
            for i in range(n_samples):
                clustser_dis[i] = self._vector_distance(X[0], X[i])
                clustser_label[i] = 0
            self.labels = clustser_label
            self.inertia = clustser_dis
            return centers, clustser_label, clustser_dis

        while (cluster_change or now_iter < self.max_iter):
            now_iter += 1
            cluster_change = False

            vis = {}
            for i in range(self.n_clusters):
                dis_pair = []
                for j in range(n_samples):
                    if j in vis:
                        continue
                    dis_pair.append((self._vector_distance(centers[i], X[j]), j))
                cnt = 0
                for _ in range(len(dis_pair)):
                    node = heapq.heappop(dis_pair)
                    vis[node[1]] = True
                    cnt += 1
                    clustser_dis[node[1]] = node[0]**2
                    if (clustser_label[node[1]] != i):
                        cluster_change = True
                    clustser_label[node[1]] = i
                    if (cnt >= centers_max_num):
                        break
            '''
            for i in range(self.n_clusters):
                pq = PriorityQueue()
                for j in range(n_samples):
                    if j in vis:
                        continue
                    pq.put((self._vector_distance(centers[i], X[j]), j))
                cnt = 0
                while ((not pq.empty()) and cnt < centers_max_num):
                    node = pq.get()
                    vis[node[1]] = True
                    cnt += 1
                    clustser_dis[node[1]] = node[0]**2
                    if (clustser_label[node[1]] != i):
                        cluster_change = True
                    clustser_label[node[1]] = i
            '''
            '''
            for i in range(n_samples):
                min_dist = np.inf
                min_index = -1
                for j in range(self.n_clusters):
                    if (max_cluster[j] >= centers_max_num):
                        continue
                    distance = self._vector_distance(centers[j], X[i])
                    if (distance < min_dist):
                        min_dist = distance
                        min_index = j
                        max_cluster[j] += 1

                if (min_index != -1 and clustser_label[i] != min_index):
                    cluster_change = True
                    clustser_label[i] = min_index
                    clustser_dis[i] = min_dist**2
            '''

            old_centers = centers.copy()
            max_shift = 0

            for i in range(self.n_clusters):
                points_in_cluster = X[clustser_label[:] == i]
                centers[i, :] = np.mean(points_in_cluster, axis=0)
                shift_ = self._vector_distance(centers[i], old_centers[i])
                max_shift = max(max_shift, shift_)
            
            if (max_shift < self.tol):
                break
            
        self.n_iter_ = now_iter
        self.cluster_centers = centers
        self.labels = clustser_label[:]
        self.inertia = clustser_dis[:]
        return centers, clustser_label, clustser_dis
    
    def predict(self, X):
        n_samples = X.shape[0]
        labels = np.full(n_samples, -1, np.int32)
        distance_net = np.zeros((n_samples, self.n_clusters))
        for i in range(n_samples):
            for j in range(self.n_clusters):
                distance_net[i, j] = self._vector_distance(X[i], self.cluster_centers[j])
            labels[i] = np.argmin(distance_net[i, :])
        return labels