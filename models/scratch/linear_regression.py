import numpy as np
import joblib

class LinearRegressionScratch:
    """
    Linear Regression using gradient descent.
    We compute predictions as y = Xw + b,
    then update weights by moving in the direction
    that reduces the Mean Squared Error cost.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias    -= self.lr * db

            loss = (1 / n_samples) * np.sum((y_pred - y) ** 2)
            self.loss_history.append(loss)

    def predict(self, X):
        X = np.array(X)
        return np.dot(X, self.weights) + self.bias
    
    @classmethod
    def load_from_weights(cls, weights_path, bias_path):
        """Load a trained model from saved weights and bias"""
        model = cls()
        model.weights = joblib.load(weights_path)
        model.bias    = joblib.load(bias_path)
        return model
