import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, x_size, y_size, layers_size, n=1e-3):
        self.x_size = x_size
        self.layers_num = len(layers_size)
        self.y_size = y_size
        self.l_shape = []
        self.l_shape.append(x_size)
        self.n = n

        self.layers = []

        for i in layers_size:
            self.layers.append(np.zeros(i))
            self.l_shape.append(i)

        self.l_shape.append(y_size)
        self.l_shape = np.asarray(self.l_shape)
        self.layers = np.asarray(self.layers)

    def init_weights(self, seed=None):
        self.weights = []
        if(seed != None):
            np.random.seed(seed)
        for i in range(len(self.l_shape)-1):
            self.weights.append(np.random.rand(self.l_shape[i+1], self.l_shape[i]))
        self.weights = np.asarray(self.weights)
    
    def init_i_and_y(self):
        self.I = []
        self.Y = []
        for i in self.l_shape:
            self.I.append(np.zeros(i))
            self.Y.append(np.zeros(i))
        del self.I[0]
        del self.Y[0]
        self.I = np.asarray(self.I)
        self.Y = np.asarray(self.Y)

    def forward(self, x):
        self.x = x
        
        self.I[0] = np.dot(self.weights[0], x)
        self.Y[0] = np.tanh(self.I[0])

        for i in range(self.layers_num):
            self.I[i+1] = np.dot(self.weights[i+1], self.Y[i])
            self.Y[i+1] = np.tanh(self.I[i+1])
    
    # Backward calcula o delta e atualiza os pesos de tras pra frente
    def backward(self, d):
        self.delta = []
        self.d = d

        for i in self.l_shape:
            self.delta.append(np.zeros(i))
        del self.delta[0]

        # Calcule backward for last layer
        # self.delta[-1] = (self.d - self.Y[-1]) * 1/np.cosh(self.I[-1])**2
        self.delta[-1] = (self.d - self.Y[-1]) * 2/(np.exp(self.I[-1]) + np.exp(-self.I[-1]))

        for i in range(len(self.weights[-1])):
            self.weights[-1][i] = self.weights[-1][i] + self.n * self.delta[-1][i] * self.Y[-2]

        # Calcule backward for hidden layers
        for i in range(1, self.layers_num):
            # self.delta[-1-i] = -np.dot(self.delta[-i], self.weights[-i]) * 1/np.cosh(self.I[-1-i])**2
            self.delta[-1-i] = -np.dot(self.delta[-i], self.weights[-i]) * 2/(np.exp(self.I[-1-i]) + np.exp(-self.I[-1-i]))

            for j in range(1, len(self.weights[-1-i])):
                self.weights[-1-i][j] = self.weights[-1-i][j] + self.n * self.delta[-1-i][j] * self.Y[-2-i]
        
        # Calcule backward for first layer
        self.delta[0] = -np.dot(self.delta[1], self.weights[1]) * 2/(np.exp(self.I[0]) + np.exp(-self.I[0]))

        for i in range(len(self.weights[0])):
            self.weights[0][i] = self.weights[0][i] + self.n * self.delta[0][i] * self.x

        
    def get_error(self):
        return (np.sum(self.d - self.Y[-1])**2)/2

    def get_output(self):
        return self.Y[-1]

    def get_layer_shape(self):
        return self.l_shape

    def get_weights(self):
        return self.weights
    
    def train(self, xinput, output, epsilon, seed=None):
        self.xinput = xinput
        self.output = output
        input_size = len(self.xinput)
        self.init_weights(seed)
        self.init_i_and_y()
        Em = 0
        Em_ant = 0
        delta_e = 1
        while delta_e >= epsilon:
            Em_ant = Em
            Em = 0
            for k in range(input_size):
                self.forward(self.xinput[k])
                self.backward(self.output[k])
                Em += self.get_error()
            Em /= input_size
            delta_e = np.abs(Em - Em_ant)

    def test(self, xinput, y_control):
        errors = 0
        for k in range(len(xinput)):
            self.forward(xinput[k])
            y_calc = self.get_output()

            if(y_calc > 0):
                y_calc = 1
            else:
                y_calc = 0

            print("Y calculado = {} | Y correto = {}".format(y_calc, y_control[k]))
            if(y_calc != y_control[k]):
                errors += 1

        erro = errors/len(y_control)*100
        print("Porcentagem de acerto = {:.2f}".format(100 - erro))
            
        
            
        


