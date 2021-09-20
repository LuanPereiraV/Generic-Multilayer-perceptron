from perceptron import Perceptron
import numpy as np

def main():

    l = [20, 20]

    file = open('input.txt', 'r')
    lines = file.readlines()
    x = []
    y = []
    input_file = []

    for line in lines:
        input_file.append([float(i) for i in line.split(',')[0:5]])

    input_file = np.asarray(input_file)
    np.random.shuffle(input_file)

    train_percent = 75
    number_train = int(train_percent/100 * len(input_file))

    _train = input_file[0:number_train]
    _test = input_file[number_train:]
    x_train = _train[:, 0:4]
    y_train = _train[:, 4]
    x_test = _test[:, 0:4]
    y_test = _test[:, 4]


    p = Perceptron(4, 1, l, n=0.001)
    p.train(x_train, y_train, epsilon=1e-4)
    p.test(x_test, y_test)
    

if(__name__ == '__main__'):
    main()
