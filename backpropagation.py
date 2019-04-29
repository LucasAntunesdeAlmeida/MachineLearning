import argparse
from _includes.readfile import *

class neuron:
    def __init__(self):
        pass

class network:
    def __init__(self, first, second, third):
        self.neurons = [
            [neuron() for i in range(first)],
            [neuron() for i in range(second)],
            [neuron() for i in range(third)]
        ]

class backpropagation:
    def __init__(self, first, second, third):
        self.network = network(first, second, third)
    
    def training(self):
        for i in range(len(self.network.neurons)):
            for j in range(len(self.network.neurons[i])):
                pass

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
    matrix = readfile(args.input)

    backprop = backpropagation(args.first, args.second, args.third)
    backprop.training()
