import numpy as np

def readfile(fileName):
    f = open(fileName, 'r')
    lines = f.readlines()
    matrix = []
    for i in lines:
        matrix.append(np.array(i.split()).astype(np.float32))
    return matrix

def getColumn(matrix, column):
    return np.array([i[column] for i in matrix]).astype(np.float32)
        
