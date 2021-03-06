import numpy as np
from _includes.readfile import readfile

class Neuron:
    def __init__(self,inputs):
        self.w = np.random.rand(inputs)
        self.y = np.float32(0.0)
        self.sigma = np.float32(0.0)
        self.x = []

inputLayer = [Neuron(15) for i in range(15)]
outputLayer = [Neuron(15) for i in range(4)]

def f(x):
    return np.float64(1.0/(1.0+np.exp(float(-x))))

def fdx(x):
    return x*(1.0-x)

def update(neuron):
    alpha = 1
    e = fdx(neuron.y)
    for i in range(len(neuron.w)):
        neuron.w[i] += alpha * neuron.sigma * e * neuron.x[i]
    return neuron.w

def first(x, i):
    # Presentation of inputs
    for neuron in inputLayer:
        neuron.y = f(np.dot(neuron.w, x[i]))
        neuron.x = x[i]
    
    # inputLayer for outputLayer
    inputData = []
    for output in inputLayer:
        inputData.append(output.y)
    for neuron in outputLayer:
        neuron.y = f(np.dot(neuron.w, inputData)) 
        neuron.x = inputData

def second(x, i):
    # Backpropagation of error
    # outputLayer for inputLayer
    for j in range(len(inputLayer)):
        inputDataW = []
        inputDataSigma = []
        for output in outputLayer:
            inputDataW.append(output.w[j])
            inputDataSigma.append(output.sigma)
        inputLayer[j].sigma = np.dot(inputDataW, inputDataSigma)

def third(x, i):
    # inputLayer
    for neuron in inputLayer:
        neuron.w = update(neuron)
    # outputLayer
    for neuron in outputLayer:
        neuron.w = update(neuron)

def showInput(input):
    for i in range(5):
        for j in range(3):
            print(int(input[(i*3)+j]), end = "")
        print()

def test(x):
    for i in range(len(x)):
        # Presentation of inputs
        for neuron in inputLayer: 
            neuron.y = f(np.dot(neuron.w, x[i]))
        # inputLayer for outputLayer
        inputData = []
        for output in inputLayer:
            inputData.append(output.y)
        for neuron in outputLayer:
            neuron.y = f(np.dot(neuron.w, inputData))
        print("\nInput : {0}".format(i))
        showInput(x[i])
        print("Output: ", end = "")
        for j in outputLayer:
            print(j.y, end=" ")
        print()

def backpropagation(matrix):
    # Separating the dataset into x and y
    x = [i[:-4] for i in matrix]
    y = [i[-4:] for i in matrix]
    
    tmax = 1000
    t = 0

    while(t < tmax):
        t += 1
        for i in range(len(x)):
            # First step
            first(x, i)
            # Error Compute (Wanted - Got)
            for j in range(len(outputLayer)):
                outputLayer[j].sigma = y[i][j] - outputLayer[j].y
            # Second step
            second(x, i)
            # Third Step
            third(x, i)

    test(x)

if __name__ == "__main__":
    backpropagation(readfile("_inputs/backpropagation-numbers.txt"))