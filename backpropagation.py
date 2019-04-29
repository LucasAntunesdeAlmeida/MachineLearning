import argparse

def arguments():
    parser = argparse.ArgumentParser(description='Implementation of a back propagation algorithm')
    parser.add_argument("-f", "--first", default=10, type=int, help="Number of neurons in the first layer")
    parser.add_argument("-s", "--second", default=15, type=int, help="Number of neurons in the second layer")
    parser.add_argument("-t", "--third", default=4, type=int, help="Number of neurons in the third layer")
    parser.add_argument("-v", "--version", action='version', version='backpropagation v1.0')
    return parser.parse_args()

if __name__ == "__main__":
    print(arguments())