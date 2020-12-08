import numpy as np
from perceptron import Perceptron

tests = []
tests.append(np.array([0,0]))
tests.append(np.array([0,1]))
tests.append(np.array([1,0]))
tests.append(np.array([1,1]))


labels = np.array([0,1,1,1])

perceptron = Perceptron(2, 10)

perceptron.train(tests, labels)

inputs = np.array([1,1])
print(perceptron.predict(inputs))

inputs = np.array([0,1])
print(perceptron.predict(inputs))

inputs = np.array([1,0])
print(perceptron.predict(inputs))