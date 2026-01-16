# logistic_regression
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000, lambda_param=0.1):
        self.lr = lr
        self.n_iters = n_iters
        self.lambda_param = lambda_param  # Regularization strength
        self.weights = None
        self.bias = None
        self.losses = []

    def _sigmoid(self, x):
        # Numerically stable sigmoid to prevent overflow
        return np.where(x >= 0, 
                        1 / (1 + np.exp(-x)), 
                        np.exp(x) / (1 + np.exp(x)))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.losses = []

        for i in range(self.n_iters):
            # 1. Forward Pass (Linear + Activation)
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(linear_pred)

            # 2. Compute Gradients (including L2 regularization for weights)
            dw = (1 / n_samples) * (np.dot(X.T, (predictions - y)) + self.lambda_param * self.weights)
            db = (1 / n_samples) * np.sum(predictions - y)

            # 3. Update Parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # Optional: Track loss (Binary Cross Entropy)
            if i % 100 == 0:
                loss = -np.mean(y * np.log(predictions + 1e-15) + (1 - y) * np.log(1 - predictions + 1e-15))
                self.losses.append(loss)

    def predict_proba(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_pred)

    def predict(self, X):
        y_proba = self.predict_proba(X)
        return (y_proba > 0.5).astype(int)

# --- Demo Code ---
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # 1. Generate synthetic binary classification data
    X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, 
                               n_clusters_per_class=1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # 2. Train the model
    model = LogisticRegression(lr=0.1, n_iters=1000)
    model.fit(X_train, y_train)

    # 3. Evaluate
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # 4. Visualize the Decision Boundary
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.5, label="Data points")
    
    # Calculate the hyperplane: w1*x1 + w2*x2 + b = 0  => x2 = -(w1*x1 + b) / w2
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x1_vals = np.linspace(x1_min, x1_max, 100)
    x2_vals = -(model.weights[0] * x1_vals + model.bias) / model.weights[1]
    
    plt.plot(x1_vals, x2_vals, color='black', linewidth=2, label="Decision Boundary")
    plt.title("Logistic Regression: Decision Boundary")
    plt.legend()
    plt.show()
