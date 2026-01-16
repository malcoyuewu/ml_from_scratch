import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k=5, max_iters=200, plot_steps=False):
        self.k = k
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        
        # Initializing storage
        self.clusters = [[] for _ in range(self.k)]
        self.centroids = []

    def dist(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2)**2))

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape
        
        # 1. Initialize centroids
        self.centroids = self._initial_centroids(X)

        for i in range(self.max_iters):
            # 2. Assign samples to closest centroids
            self.clusters = self._create_clusters(self.centroids)
            
            if self.plot_steps:
                self.plot()

            # 3. Update centroids
            old_centroids = self.centroids
            self.centroids = self._update_centroids(self.clusters)
            
            # 4. Check for convergence
            if self._is_converged(old_centroids, self.centroids):
                print(f"Converged at iteration {i}")
                break
                
        return self._get_cluster_labels(self.clusters)

    def _initial_centroids(self, X, plus = False):
        if plus: # kmeans plus
            n_samples, n_features = X.shape
            # 1. Randomly select the first centroid from the data points
            centroids = [X[np.random.randint(n_samples)]]
    
            # 2. Select the remaining K-1 centroids
            for _ in range(1, self.k):
                # Calculate the squared distance from each point to the nearest existing centroid
                dists = np.array([min([np.sum((x - c)**2) for c in centroids]) for x in X])
                
                # Calculate probabilities: P(x) = D(x)^2 / sum(D(x)^2)
                probs = dists / dists.sum()
                cumulative_probs = np.cumsum(probs)
                r = np.random.rand()
    
                # Pick the next centroid based on the probability distribution
                for idx, p in enumerate(cumulative_probs):
                    if r < p:
                        centroids.append(X[idx])
                        break
                    
            return np.array(centroids)
            
        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)
        # Generate random points within the bounding box of the data
        return np.random.uniform(X_min, X_max, size=(self.k, self.n_features))

        

    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.k)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        distances = [self.dist(sample, c) for c in centroids]
        return np.argmin(distances)

    def _update_centroids(self, clusters):
        centroids = np.zeros((self.k, self.n_features))
        for idx, cluster in enumerate(clusters):
            if len(cluster) > 0:
                centroids[idx] = np.mean(self.X[cluster], axis=0)
            else:
                # If cluster is empty, keep the old centroid or re-randomize
                centroids[idx] = self.X[np.random.choice(range(self.n_samples))]
        return centroids

    def _is_converged(self, old, new):
        # Using np.allclose to handle floating point precision
        return np.allclose(old, new)

    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    def plot(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, cluster in enumerate(self.clusters):
            points = self.X[cluster].T
            ax.scatter(*points)
        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)
        plt.show()

# Example Usage
if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(centers=3, n_samples=300, random_state=42)
    model = KMeans(k=3, plot_steps=False)
    labels = model.predict(X)
    model.plot()
