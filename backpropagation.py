import argparse
from _includes.readfile import *

class neuron:
    def __init__(self, inputs):
        self.w = np.random.rand(inputs) 
        self.y = np.float32(0.0)
    
    def h(self, inputs):
        pass

class network:
    def __init__(self, inputs, first, second, third):
        self.neurons = [
            [neuron(inputs) for i in range(first)],
            [neuron(first) for i in range(second)],
            [neuron(second) for i in range(third)]
        ]

class backpropagation:
    def __init__(self, first, second, third, matrix):
        self.dataset = matrix
        self.datasetT = [getColumn(matrix, i) for i in range(len(matrix[0]))]
        self.x = self.datasetT[:-third]
        self.y = self.datasetT[-third:]
        self.maxTime = 10000
        self.network = network(len(self.x), first, second, third)

    def error(self, time):
        return (self.maxTime > time)

    def firstStage(self):
        pass

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
    backprop = backpropagation(args.first, args.second, args.third, readfile(args.input))
    backprop.training()
