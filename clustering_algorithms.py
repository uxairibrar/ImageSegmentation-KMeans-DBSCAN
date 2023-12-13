# Foundations of Data Mining - Practical Task 1
# Version 2.0 (2023-11-02)
###############################################
# Template for a custom clustering library.
# Classes are partially compatible to scikit-learn.
# Aside from check_array, do not import functions from scikit-learn, tensorflow, keras or related libraries!
# Do not change the signatures of the given functions or the class names!

import numpy as np
from sklearn.utils import check_array
from scipy.spatial.distance import cdist

class CustomKMeans:
    def __init__(self, n_clusters=8, max_iter=300, random_state=None):
        """
        Creates an instance of CustomKMeans.
        :param n_clusters: Amount of target clusters (=k).
        :param max_iter: Maximum amount of iterations before the fitting stops (optional).
        :param random_state: Initialization for randomizer (optional).
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X: np.ndarray, y=None):
        """
        This is the main clustering method of the CustomKMeans class, which means that this is one of the methods you
        will have to complete/implement. The method performs the clustering on vectors given in X. It is important that
        this method saves the centroids in "self.cluster_centers_" and the labels (=mapping of vectors to clusters) in
        the "self.labels_" attribute! As long as it does this, you may change the content of this method completely
        and/or encapsulate the necessary mechanisms in additional functions.
        :param X: Array that contains the input feature vectors
        :param y: Unused
        :return: Returns the clustering object itself.
        """
        X = check_array(X, accept_sparse='csr') # Validating the data recieved
        random_generator = np.random.default_rng(self.random_state) # Initilizing the cluster through kmean
        self.cluster_centers_ = X[random_generator.choice(
            len(X), size=self.n_clusters, replace=False)]
        for _ in range(self.max_iter): # Updating Clusters and assignments 
            
            # Step 1: Assign each data point to the nearest cluster
            distances = np.linalg.norm(
                X[:, np.newaxis] - self.cluster_centers_, axis=2)  # Euclidean distance
            self.labels_ = np.argmin(distances, axis=1)

            # Step 2: Update the cluster centers
            new_centers = np.array([X[self.labels_ == i].mean(
                axis=0) for i in range(self.n_clusters)])

            # Check for convergence
            if np.all(new_centers == self.cluster_centers_):
                break

            self.cluster_centers_ = new_centers

        return self

    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Calls fit() and immediately returns the labels. See fit() for parameter information.
        """
        self.fit(X)
        return self.labels_



class CustomDBSCAN:
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean'):
        """
        Creates an instance of CustomDBSCAN.
        :param min_samples: Equivalent to minPts. Minimum amount of neighbors of a core object.
        :param eps: Short for epsilon. Radius of considered circle around a possible core object.
        :param metric: Used metric for measuring distances (optional).
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.labels_ = None

    def find_neighbors(self, data_matrix, target_index):
        distances = cdist(data_matrix[[target_index]], data_matrix, metric=self.metric)[0]
        neighbors_indices = np.where(distances <= self.eps)[0]
        return neighbors_indices

    def expand_cluster(self, start_index, start_neighbors, cluster_label, point_labels, data_matrix):
        point_labels[start_index] = cluster_label
        i = 0
        while i < len(start_neighbors):
            current_point = start_neighbors[i]
            if point_labels[current_point] == -1:
                point_labels[current_point] = cluster_label # Assigning the cluster label to the current point
                current_neighbors = self.find_neighbors(data_matrix, current_point) # Check neighbopring labels of this point
                if len(current_neighbors) >= self.min_samples: # Check if the current point is a core point
                    # Explanding the cluster by ading current neighbors
                    start_neighbors = np.concatenate((start_neighbors, current_neighbors)) 
            i += 1

     
    def fit(self, X: np.ndarray, y=None):
        """
        This is the main clustering method of the CustomDBSCAN class, which means that this is one of the methods you
        will have to complete/implement. The method performs the clustering on vectors given in X. It is important that
        this method saves the determined labels (=mapping of vectors to clusters) in the "self.labels_" attribute! As
        long as it does this, you may change the content of this method completely and/or encapsulate the necessary
        mechanisms in additional functions.
        :param X: Array that contains the input feature vectors
        :param y: Unused
        :return: Returns the clustering object itself.
        """
        X = check_array(X, accept_sparse='csr') # Validating the data recieved
        n_samples = X.shape[0]
        labels = -np.ones(n_samples)  # Assign noise to the labels by default, -1 is the noise
        cluster_label = 0 
        
        for i in range(n_samples): # Iterating through each sample
            if labels[i] != -1:  # Skip already assigned points
                continue
            neighbors = self.find_neighbors(X, i) # Getting neighbors for the current point
            if len(neighbors) < self.min_samples: # Assigning noise because it is has less number of neighbors
                labels[i] = -1
            else:
                cluster_label += 1
                # Assign Cluster and expand it, because the number of neigbors are more than the minimum
                self.expand_cluster(i, neighbors, cluster_label,labels,X)  

        self.labels_ = labels
        return self

    def fit_predict(self, X: np.ndarray, y=None) -> np.ndarray:
        """
        Calls fit() and immediately returns the labels. See fit() for parameter information.
        """
        self.fit(X)
        return self.labels_
