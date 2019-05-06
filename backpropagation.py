import argparse
from _includes.readfile import *

class Neuron:
    def __init__(self, inputs):
        self.w = np.random.rand(inputs) 
        self.y = np.float32(0.0)
    
    def h(self, neuronInput):
        # print(self.w)
        # print(len(neuronInput))
        self.y = np.float32(0.0)

class Network:
    def __init__(self, inputs, first, second, third):
        self.neurons = [
            [Neuron(inputs) for i in range(first)],
            [Neuron(first) for i in range(second)],
            [Neuron(second) for i in range(third)]
        ]

class Backpropagation:
    def __init__(self, first, second, third, matrix):
        self.dataset = matrix
        self.datasetT = [getColumn(matrix, i) for i in range(len(matrix[0]))]
        self.x = self.datasetT[:-third]
        self.y = self.datasetT[-third:]
        self.maxTime = 1
        self.network = Network(len(self.x), first, second, third)
        self.delta = self.y[:]

    def error(self, time):
        return (self.maxTime > time)

    def firstStage(self):
        # initialization that guarantees the input value of the first layer
        neuronInput = self.x[:]
        # presentation of the inputs to the network
        # and propagation of the outputs to the network
        for network in self.network.neurons:
            temp = []
            # passes to the current input to the current neuron 
            # and saves the output of the neuron
            for neuron in network:
                neuron.h(neuronInput)
                temp.append(neuron.y)
            # copy of the output values of each layer
            neuronInput = temp[:]
        # error computation (desired-obtained)
        for i in range(len(self.network.neurons[-1])):
            self.delta[i] = self.y[i] - self.network.neurons[-1][i].y
            # print(self.delta[i])
            


    def secondStage(self):
        pass
    
    def thirdStage(self):
        pass

    def training(self):
        time = 0
 
        while(self.error(time)):
            time += 1
            self.firstStage()
            self.secondStage()
            self.thirdStage()

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
