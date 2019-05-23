import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt
from _includes.readfile import readfile

class knn():
    def __init__(self, input, k):
        self.k = k
        self.data = []
        self.datatype = []
        self.colors = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#8B4513", "#2ecc71"]
        self.symbols = ["o", "v", "s", "p", "*", "X"]

        for data in input:
            self.data.append(data[:-1])
            self.datatype.append(data[-1:])

        self.plot(data, input)

    def minkowski(self, i, coord, p):
        ac = 0
        for j in range(len(self.data[i])):
            ac += ((coord[j] - self.data[i][j])**p)
        return (ac**(1/p))
    
    def distance(self, input):
        dist = []
        for i in range(len(self.data)):
            dist.append([self.minkowski(i, input, 2),i])
        dist = sorted(dist, key=itemgetter(0), reverse=False)
        return dist

    def setTypes(self):
        self.type = {}
        for data in self.datatype:
            self.type[int(data)] = 0

    def classification(self, data, input):
        self.setTypes()
        for i in range(self.k):
            self.type[self.datatype[data[i][1]][0]] += 1
    
    def plot(self, data, input):
        for i in range(len(self.data)):
            plt.plot(self.data[i][0], self.data[i][1], self.symbols[int(self.datatype[i])], color = self.colors[int(self.datatype[i])])

    def group(self, input):
        self.classification(self.distance(input), input)
        self.type = sorted(self.type.items(), key=itemgetter(1), reverse=True)
        index = self.type[0][0]
        plt.plot(input[0], input[1], self.symbols[index], color = "#bd0505")
    
    def show(self):
        plt.show()
            
if __name__ == "__main__":
    KNN = knn(readfile("_inputs/knn.txt"), 7)  
    KNN.group([2,2])
    KNN.group([5,6])
    KNN.show()