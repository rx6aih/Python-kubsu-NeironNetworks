import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(x, 0)

X = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
y = np.array([0, 0, 1, 1])

class NeuralNetwork:
    def __init__(self):
        self.weights = np.random.randn(2)
        self.bias = np.random.randn()

    def forward(self, x):
        return relu(np.dot(x, self.weights) + self.bias)

    def loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def train(self, X, y, epochs=1000, lr=0.01):
        for _ in range(epochs):
            y_pred = self.forward(X)
            loss = self.loss(y_pred, y)
            self.weights -= lr * np.dot(X.T, y_pred - y)
            self.bias -= lr * np.mean(y_pred - y)

model = NeuralNetwork()

model.train(X, y)

y_pred = model.forward(X)

x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100))
Z = model.forward(np.c_[xx1.ravel(), xx2.ravel()]).reshape(xx1.shape)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
plt.contour(xx1, xx2, Z, levels=[0.5], colors='black')
plt.show()