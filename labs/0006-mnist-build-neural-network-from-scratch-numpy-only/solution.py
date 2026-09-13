import numpy as np

class NeuralNetwork:
    '''
    Build a neural network from scratch using only NumPy.
    Architecture: 784 → 128 (ReLU) → 10 (Softmax)
    '''

    def __init__(self, input_size=784, hidden_size=128, output_size=10, lr=0.01):
        self.lr = lr

        # He initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.b1 = np.zeros(hidden_size)

        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
        self.b2 = np.zeros(output_size)

    def forward(self, X):
        # First linear layer
        z1 = X @ self.W1 + self.b1

        # ReLU
        a1 = np.maximum(z1, 0)

        # Second linear layer
        z2 = a1 @ self.W2 + self.b2

        # Stable softmax
        mx = np.max(z2, axis=-1, keepdims=True)
        exp_z2 = np.exp(z2 - mx)
        probs = exp_z2 / np.sum(exp_z2, axis=-1, keepdims=True)

        # Cache for backward
        self.z1 = z1
        self.a1 = a1

        return probs

    def backward(self, X, y, probs):
        N = X.shape[0]

        # Cross-entropy loss
        correct_probs = probs[np.arange(N), y]
        loss = -np.mean(np.log(correct_probs + 1e-12))

        # Softmax + cross-entropy
        dz2 = probs.copy()
        dz2[np.arange(N), y] -= 1
        dz2 /= N

        # Layer 2
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        da1 = dz2 @ self.W2.T

        # ReLU
        dz1 = da1 * (self.z1 > 0)

        # Layer 1
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)

        # Gradient descent
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        return loss

    def train_step(self, X, y):
        probs = self.forward(X)
        return self.backward(X, y, probs)

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)