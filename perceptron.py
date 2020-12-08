import numpy as np

class Perceptron(object):
    def __init__(self, numberInputs,epochs=100, learningRate = 0.01):
        self.epochs = epochs
        self.learningRate = learningRate
        self.weights = np.zeros(numberInputs + 1)

    def predict(self, inputs):
        sum = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if sum > 0:
            return 1 #activate
        else:
            return 0

    def train(self, trainingInputs, labels):
        for value in range(self.epochs):
            for inputs, label in zip(trainingInputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learningRate*(label-prediction)*inputs
                self.weights[0] += self.learningRate*(label-prediction)