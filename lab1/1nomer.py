from random import random

nu = 0.5

class Neuron:
    def __init__(self):
        self.w = [random(), random()]
        divider = (self.w[0] ** 2 + self.w[1] ** 2) ** (0.5)
        self.w[0] /= divider
        self.w[1] /= divider

    def calculate(self, x):
        return self.w[0]*x[0] + self.w[1]*x[1]

    def recalculate(self, x):
        self.w[0] += nu*(x[0] - self.w[0])
        self.w[1] += nu*(x[1] - self.w[1])

class NeuralNetwork:
    def __init__(self) -> None:
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
        self.neurons = [Neuron() for i in range(4)]
    
    def __str__(self):
        s = ''
        for neuron in self.neurons:
            s += str(neuron.w) + '\n'
        return s
    
    def start(self):
        u = [0] * 4
        wins = [0] * 4
        for i in range(len(self.x)):
            for j in range(4):
                    u[j] = self.neurons[j].calculate(self.x[i])
            j = u.index(max(u))
            self.neurons[j].recalculate(self.x[i])
            wins[j] += 1
        print('wins:', wins)

nn = NeuralNetwork()
print(nn)
print("----------------")
nn.start()
print(nn)