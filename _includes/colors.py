import numpy as np

def genHexaColor():
    color = ''
    symbols = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    for i in range(6):
        color += symbols[np.random.randint(16)]
    return '#'+color

def genHexaColorList(num):
    colors = []
    for i in range(num):
        colors.append(genHexaColor())
    return colors