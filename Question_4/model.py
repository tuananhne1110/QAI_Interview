import numpy as np
import h5py

class LogisticRegressionAPI:
    def __init__(self, model_file):
        self.model_file = model_file
        self.w = self.load_model()

    def load_model(self):
        with h5py.File(self.model_file, 'r') as f:
            return np.array(f['weights'])

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def classify(self, X):
        y_hat = self.sigmoid(np.matmul(X, self.w))
        labels = np.argmax(y_hat, axis=1)
        return labels.reshape(-1, 1)
