import argparse
import numpy as np
from _includes.readfile import *

class Neuron:
    def __init__(self, inputs):
        self.w = np.random.rand(inputs)
        self.y = np.float32(0.0)
        self.sigma = np.float32(0.0)
        self.alpha = np.float32(0.01)
        self.e = np.float32(0.0)
    
    def f(self, e):
        return 1/(1+np.exp(-e))
    
    def fdx(self, e):
        return (1-np.exp(-e)/1+np.exp(-e))

    def h(self, neuronInput):
        self.e = np.dot(neuronInput, self.w)
        self.y = self.f(self.e)
    
    def setSigma(self, w, sigma):
        self.sigma = np.dot(w,sigma)
    
    def setWeight(self):
        for i in range(len(self.w)):
            self.w[i] += (self.alpha*self.sigma*self.fdx(self.e))

class Network:
    def __init__(self, inputs, first, second, third):
        self.neurons = [
            [Neuron(inputs) for i in range(first)],
            [Neuron(first) for i in range(second)],
            [Neuron(second) for i in range(third)]
        ]
    
    def checkLayer(self, layer):
        if layer >= 0 and layer < len(self.neurons):
            return True
        else:
            print("nonexistent layer: "+str(layer))
            return False

    def getY(self, layer):
        if self.checkLayer(layer):
            return [neuron.y for neuron in self.neurons[layer]]
    
    def getW(self, layer, i):
        if self.checkLayer(layer):
            return [neuron.w[i] for neuron in self.neurons[layer]]
    
    def getSigma(self, layer):
        if self.checkLayer(layer):
            return [neuron.sigma for neuron in self.neurons[layer]]

class Backpropagation:
    def __init__(self, first, second, third, matrix):
        self.dataset = matrix
        self.datasetT = [getColumn(matrix, i) for i in range(len(matrix[0]))]
        self.x = np.array(self.datasetT[:-third]).T
        self.y = self.datasetT[-third:]
        self.maxTime = 1000
        self.network = Network(len(matrix[0])-third, first, second, third)

    def error(self, time):
        return (self.maxTime > time)

    def firstStage(self, j):
        # presentation of the inputs to the network
        # and propagation of the outputs to the network
        neuronInput = self.x[j]    
        for i in range(len(self.network.neurons)):
            for neuron in self.network.neurons[i]:
                neuron.h(neuronInput)
            neuronInput = self.network.getY(i)
        for i in range(len(self.network.neurons[-1])):
            self.network.neurons[-1][i].sigma = self.y[i][j] - self.network.neurons[-1][i].y
        
    def secondStage(self):
        # each neuron calculates its sigma error based on the errors and weights of the next layer
        for i in reversed(range(len(self.network.neurons)-1)):
            for j in range(len(self.network.neurons[i])):
                self.network.neurons[i][j].setSigma(self.network.getW(i+1,j),self.network.getSigma(i+1))

    def thirdStage(self):
        # since all the neurons already have the necessary data, 
        # it runs through the entire network and make each neuron 
        # update its respective weight
        for network in self.network.neurons:
            for neuron in network:
                neuron.setWeight()
        

    def training(self):
        time = 0
        while(self.error(time)):
            time += 1
            for i in range(len(self.x)):
                self.firstStage(i)
                self.secondStage()
                self.thirdStage()
        
        for respost in self.network.neurons[-1]:
            print(respost.y,end=" ")
        print()

def arguments():
    parser = argparse.ArgumentParser(description='Implementation of a back propagation algorithm')
    parser.add_argument("-f", "--first", default=10, type=int, help="Number of neurons in the first layer")
    parser.add_argument("-s", "--second", default=15, type=int, help="Number of neurons in the second layer")
    parser.add_argument("-t", "--third", default=4, type=int, help="Number of neurons in the third layer")
    parser.add_argument("-i", "--input", default='_inputs/backpropagation-numbers.txt', type=str, help="Archive for Network Training")
    parser.add_argument("-v", "--version", action='version', version='backpropagation v1.0')
    return parser.parse_args()

if __name__ == "__main__":
    args = arguments()
    backprop = Backpropagation(args.first, args.second, args.third, readfile(args.input))
    backprop.training()
