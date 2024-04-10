import numpy as np

class FuzzyNeuralNetwork:
    def __init__(self, num_inputs, num_outputs, num_rules, num_membership_functions):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_rules = num_rules
        self.num_membership_functions = num_membership_functions
        self.weights = np.random.rand(num_rules, num_outputs)
        self.biases = np.random.rand(num_outputs)
        self.membership_functions = [FuzzyMembershipFunction() for _ in range(num_membership_functions)]
        
    def forward_pass(self, inputs):
        membership_degrees = [mf.calculate(x) for mf, x in zip(self.membership_functions, inputs)]
        rule_activations = np.min(membership_degrees, axis=0)
        outputs = np.dot(rule_activations, self.weights) + self.biases
        return outputs

    def train(self, X, y, epochs=100, learning_rate=0.1):
        for _ in range(epochs):
            outputs = self.forward_pass(X)
            error = y - outputs
            for i in range(self.num_rules):
                self.weights[i] -= learning_rate * error * rule_activations[i] # type: ignore
            self.biases -= learning_rate * error

class FuzzyMembershipFunction:

    def __init__(self, type="gaussian", params=[0, 1]):
        self.type = type
        self.params = params

    def calculate(self, x):
        if self.type == "gaussian":
            mean, sigma = self.params
            return np.exp(-(x - mean)**2 / (2 * sigma**2))
        elif self.type == "triangular":
            a, b, c = self.params
            if x < a:
                return 0
            elif a <= x <= b:
                return (x - a) / (b - a)
            elif b < x <= c:
                return (c - x) / (c - b)
            else:
                return 0
        else:
            raise ValueError("Неизвестный тип функции принадлежности.")
network = FuzzyNeuralNetwork(2, 1, 4, 3)

network.train(X, y)

predictions = network.forward_pass(new_inputs) # type: ignore