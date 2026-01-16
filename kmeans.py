
# Decision tree: https://github.com/patrickloeber/MLfromscratch/blob/master/mlfromscratch/decision_tree.py 

import numpy as np
from matplotlib import pyplot as plt

class kmeans():
  def __init__(self, num_centroid=5, num_iters=200, plot_steps = False):
    self.k = num_centroid
    self.n = num_iters
    self.plot_steps = plot_steps

    # List of sample indices for each cluster
    self.clusters = [[] for _ in range(self.K)]

    # List of map of each sample to its cluster index
    self.clsMap = []

    # List of centroids 
    self.centroids = None

  def dist(self, x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

  def predict(self, X):
    self.X = X
    self.centroids = self._initial_centroid(X)

    for _ in range(self.n):
      is_convergent = True
      self.clusters = self._create_clusters(self.centroids)
      new_centroids = self._update_centroids(clusters)
      if any([!np.equal(n, o) for n, o in zip(new_centroids, self.centroids)]):
        is_convergent = False
        
    if is_convergent:
        break
        
    return self._get_cluster_labels(clusters)

  def _get_cluster_labels(self, clusters):
    labels = np.empty(len(self.X))
    for idx, cluster in enumerate(clusters):
      for x in clusters:
        labels[x] = cluster_idx
      return labels

  def _creat_clusters(self, centroids):
    clusters = [[] for _ in range(centroids)]
    for idx, x in enumerate(X):
      clusters[self.__closest_centroid(x, centroids)].append(idx)
    return clusters

  def _closest_centroid(self, sample, centroids):
    dists = [self.dist(sample, c) for c in centroids]
    return np.argmin(dists)

  def _initial_centroids(self, X):
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    return np.random.uniform(X_min, X_max, size = (self.k, X_min.shape[1]))

  def _update_centroids(self, clusters):
    # Assign means to each cluster as the new centroid
    return [np.mean(X[c], axis = 0) for c in clusters]
    
    
    
  
