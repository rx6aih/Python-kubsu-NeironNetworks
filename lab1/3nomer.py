from random import random

nu = 0.5

class Neuron:
    def __init__(self):
        self.w = [random(), random()]
        leng = (self.w[0] ** 2 + self.w[1] ** 2) ** (0.5)
        self.w[0] /= leng
        self.w[1] /= leng

    def calculate(self, x):
        return self.w[0]*x[0] + self.w[1]*x[1]

    def recalculate(self, x, u):
        self.w[0] += nu*x[0]*u
        self.w[1] += nu*x[1]*u

class NeuralNetwork:
    def __init__(self):
        self.x = [
            [0.97, 0.2],
            [1, 0],
            [-0.72, 0.7],
            [-0.67, 0.74],
            [-0.8, 0.6],
            [0, -1],
            [0.2, -0.97],
            [-0.3, -0.95]
        ]
        self.neurons = [Neuron() for i in range(2   )]
    
    def __str__(self):
        s = ''
        for neuron in self.neurons:
            s += str(neuron.w) + '\n'
        return s
    
    def start(self):
        u = [0] * 2
        for i in range(len(self.x)):
            for j in range(2):
                u[j] = self.neurons[j].calculate(self.x[i])
                self.neurons[j].recalculate(self.x[i], u[j])

nn = NeuralNetwork()
print(nn)
nn.start()
print(nn)