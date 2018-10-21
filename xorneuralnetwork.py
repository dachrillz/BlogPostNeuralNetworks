import math
import numpy as np
import matplotlib.pyplot as plt

##########################################################################
# Some Class Definitions
#########################################################################

class Neuron:
    def __init__(self, weights, bias, layer, previous, final = False):
        self.weights = weights
        self.bias = bias
        self.layer = layer
        self.previous = previous
        self.FINAL = final

    def calculate(self):
        prevresult = []
        for item in self.previous:
            prevresult.append(item.calculate())

        temp = sum([a*b for a,b in zip(self.weights,prevresult)])
        if self.FINAL:
            self.calculated_value = np.tanh(temp + self.bias)
        else:
            self.calculated_value = np.tanh(temp + self.bias)

        return self.calculated_value

    def relu(self,x):
        if x>0:
            return x
        return 0

    def sigmoid(self,x):
        return 1 / (1 + math.exp(-x))

class InputNeuron:
    def setvalue(self,x):
        self.x = x

    def calculate(self):
        return self.x


##########################################################################
# Here the actual fun starts,
# we create a neural network with two inner nodes called z1, and z2
# we show that this (with the correct parameters) can create a decision
# boundary such that when we run the whole network, it produces
# output values such that when they get rounded to -1, or 1, the -1
# correlates to a logic 0, and 1 to a logic 1 in the xor function.
# That is the network is expected to produce the following:
    ##########################################
    # ....x1.x2....Y....XOR..
    # -----------|-----------
    # --| 0  0   | -1 | 0   |
    # --| 0  1   | 1  | 1   |
    # --| 1  0   | 1  | 1   |
    # --| 1  1   | -1 | 0   |
    # -----------------------
    ###########################################
##########################################################################

if __name__ == '__main__':
    #construct a neural network with two inner nodes

    #input layer
    x1 = InputNeuron()
    x2 = InputNeuron()
    inputLayer = [x1,x2]

    #hidden layer
    weightshidden1 = [4,4]
    weightshidden2 = [-3,-3]
    z1 = Neuron(weightshidden1,-2,1,inputLayer)
    z2 = Neuron(weightshidden2,5,1,inputLayer)
    hiddenLayer = [z1,z2]

    #output layer
    weightsoutput = [5,5]
    y = Neuron(weightsoutput,-5,2,hiddenLayer,final=True)

    ##########################################
    #We try all the possible input combinations
    ##########################################

    # x1 and x2 are the possible inputs, and y is the expected result
    result_list = []

    #expect y=1
    x1.setvalue(0)
    x2.setvalue(0)
    result_list.append(y.calculate())

    #expect y=1
    x1.setvalue(0)
    x2.setvalue(1)
    result_list.append(y.calculate())

    #expect y=1
    x1.setvalue(1)
    x2.setvalue(0)
    result_list.append(y.calculate())

    #expect y=-1
    x1.setvalue(1)
    x2.setvalue(1)
    result_list.append(y.calculate())

    print(result_list) # [-1,1,1,-1]

    #########################################
    # Plot the result as a decision boundary
    ########################################

    x1list = np.linspace(-1,2,100)
    x2list = np.linspace(-1,2,100)
    resultlist = []
    for item in x1list:
        templist = []
        for item2 in x2list:
            x1.setvalue(item)
            x2.setvalue(item2)
            templist.append(y.calculate())
        resultlist.append(templist)

    #contour plot
    plt.imshow(resultlist, extent=[0, 5, 0, 5], origin='lower',
               cmap='RdGy', alpha=0.5)
    plt.colorbar();
    plt.show()


    #########################################################
    # We plot the intermediate values of the hidden layer
    # as (x1,x2) -> (z1,z2), because they demonstrate
    # why the neural network can solve this particular problem.
    #########################################################

    #z1
    x1list = np.linspace(-1,2,100)
    x2list = np.linspace(-1,2,100)
    resultlist = []
    for item in x1list:
        templist = []
        for item2 in x2list:
            x1.setvalue(item)
            x2.setvalue(item2)
            temp = z1.calculate()
            templist.append(temp)

        resultlist.append(templist)

    #contour plot
    plt.imshow(resultlist, extent=[0, 5, 0, 5], origin='lower',
               cmap='RdGy', alpha=0.5)
    plt.colorbar();
    plt.show()


    #z2
    x1list = np.linspace(-1,2,100)
    x2list = np.linspace(-1,2,100)
    resultlist = []
    for item in x1list:
        templist = []
        for item2 in x2list:
            x1.setvalue(item)
            x2.setvalue(item2)
            temp = z2.calculate()
            templist.append(temp)

        resultlist.append(templist)

    #contour plot
    plt.imshow(resultlist, extent=[0, 5, 0, 5], origin='lower',
               cmap='RdGy', alpha=0.5)
    plt.colorbar();
    plt.show()

    ##################################################################
    # In order to demonstrate that we need at least 2 inner neurons
    # We take a look at what happens when we only use a single neuron.
    #################################################################

    #construct a neural network with 1 inner node

    #input layer
    x1 = InputNeuron()
    x2 = InputNeuron()
    inputLayer = [x1,x2]

    #hidden layer
    weightshidden1 = [4,4]
    zsingle = Neuron(weightshidden1,-2,1,inputLayer)
    hiddenLayer = [zsingle]

    #output layer
    weightsoutput = [5]
    y = Neuron(weightsoutput,-5,2,hiddenLayer,final=True)

    ##########################################
    # Single hidden Node
    # We try all the possible input combinations
    ##########################################

    # x1 and x2 are the possible inputs, and y is the expected result
    result_list = []

    #expect y=1
    x1.setvalue(0)
    x2.setvalue(0)
    result_list.append(y.calculate())

    #expect y=1
    x1.setvalue(0)
    x2.setvalue(1)
    result_list.append(y.calculate())

    #expect y=1
    x1.setvalue(1)
    x2.setvalue(0)
    result_list.append(y.calculate())

    #expect y=-1
    x1.setvalue(1)
    x2.setvalue(1)
    result_list.append(y.calculate())

    print(result_list) # [-1,0,0,0]

    x1list = np.linspace(-1,2,100)
    x2list = np.linspace(-1,2,100)
    resultlist = []
    for item in x1list:
        templist = []
        for item2 in x2list:
            x1.setvalue(item)
            x2.setvalue(item2)
            temp = zsingle.calculate()
            templist.append(temp)

        resultlist.append(templist)

    #contour plot
    plt.imshow(resultlist, extent=[0, 5, 0, 5], origin='lower',
               cmap='RdGy', alpha=0.5)
    plt.colorbar();
    plt.show()
