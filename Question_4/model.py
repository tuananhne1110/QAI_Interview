import numpy as np
import pickle

def __init__(self, d_in, d1, d2, d_out, lr=1e-3):
        """
        Initialize the neural network.

        Args:
            d_in (int): Dimension of the input layer.
            d1 (int): Dimension of the first hidden layer.
            d2 (int): Dimension of the second hidden layer.
            d_out (int): Dimension of the output layer.
            lr (float): Learning rate for the network. Default is 1e-3.
        """
        self.d_in = d_in
        self.d1 = d1
        self.d2 = d2
        self.d_out = d_out
        self.lr = lr
        self.init_weights()

    def init_weights(self):
        """Initialize weights and biases for the network."""
        self.w1 = np.random.randn(self.d1, self.d_in)
        self.b1 = np.random.randn(self.d1, 1)

        self.w2 = np.random.randn(self.d2, self.d1)
        self.b2 = np.random.randn(self.d2, 1)

        self.w3 = np.random.randn(self.d_out, self.d2)
        self.b3 = np.random.randn(self.d_out, 1)

    def relu(self, x):
        """
        Apply ReLU activation function.

        Args:
            x (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: Output array with ReLU applied.
        """
        return np.maximum(x, 0)

    def soft_max(self, x):
        """
        Apply softmax activation function.

        Args:
            x (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: Output array with softmax applied.
        """
        x = x - np.max(x, axis=0)
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def forward(self, x, y=None):
        """
        Perform forward pass of the network.

        Args:
            x (numpy.ndarray): Input data.
            y (numpy.ndarray, optional): True labels. Default is None.
        """
        self.x = x
        if y is not None:
            self.y = y

        self.z1 = np.matmul(self.w1, self.x) + self.b1
        self.a1 = np.apply_along_axis(self.relu, 1, self.z1)

        self.z2 = np.matmul(self.w2, self.a1) + self.b2
        self.a2 = np.apply_along_axis(self.relu, 1, self.z2)

        self.z3 = np.matmul(self.w3, self.a2) + self.b3
        self.out = np.apply_along_axis(self.soft_max, 1, self.z3)

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
