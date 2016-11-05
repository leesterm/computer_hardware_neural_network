#@Leester Mei
import math
import numpy as np
class NeuralNetwork:
  def __init__(self, sizes,learning_rate):
    self.layers = len(sizes)
    self.sizes = sizes
    self.learning_rate = learning_rate
    self.biases = []
    for i in range(1,len(sizes)):
      b = []
      for j in range(sizes[i]):
        b.append(np.random.randn())
      self.biases.append(b)
    self.weights = []
    for i in range(len(sizes)-1):
      w = []
      for j in range(sizes[i]):
        w.append(np.random.randn())
      self.weights.append(w)
  #Propogate input data foward by one layer, starting with input data from layer ending in output data in layer+1   
  def forwardpropogate(self,input,layer):
    output = []
    for i in range(self.sizes[layer+1]):
      o = 0
      for j in range(len(input)):
        o += input[j]*self.weights[layer][j]
      output.append(self.sigmoid(o+self.biases[layer][i]))
    return output
  #Back propogation algorithm to adjust weights and biases based on target values and neural network outputs
  def backpropogate(self,input,output):
    O_j = net.forwardpropogate(input,0) #Output of hidden layer
    O_k = net.forwardpropogate(O_j,1) #Output of the output layer
    #Compute Errors
    delta_k = [] #Output layer Error
    for i in range(self.sizes[2]): #For each output layer node output, compute its error
      delta_k.append(O_k[i]*(1-O_k[i])*(O_k[i] - output[i]))
    delta_j = [] #Hidden Layer Error
    for i in range(self.sizes[1]): #For each hidden layer node output, computes its error
      sum = 0
      for j in range(self.sizes[2]): #For each output layer node, compute the product of the output error, delta_k, and weight from node in layer j to layer k
        sum += delta_k[j]*self.weights[1][i]
      delta_j.append(O_j[i]*(1-O_j[i])*(sum))
    #Begin backpropogation algorithm
    #Adjust Weights
    for w in range(len(self.weights)):
      for l in range(len(self.weights[w])):
        if w == 0: #Hard code cases since deltas are in seperate arrays
          self.weights[w][l] = self.weights[w][l]-(self.learning_rate)*delta_j[0]*input[l]
        elif w == 1:
          self.weights[w][l] = self.weights[w][l]-(self.learning_rate)*delta_k[0]*O_j[l]
        
    #Adjusts Biases
    for b in range(len(self.biases)):
      for l in range(len(self.biases[b])):
        if b == 0:
          self.biases[b][l] = self.biases[b][l]-(self.learning_rate)*(delta_j)[l]
        elif b == 1:
          self.biases[b][l] = self.biases[b][l]-(self.learning_rate)*(delta_k)[l]
    #End backpropogation algorithm
    
  @staticmethod
  #Our squashing/logistic/activation function given input vector z
  def sigmoid(z):
    return 1/(1+np.exp(-z))
  
net = NeuralNetwork([7,3,1],0.1)
import sys
with open(sys.argv[1],'r') as f: #Import normalized data
  for line in f: #Each line in data is 7 input values, and 1 target output value, delimited by commas
    input = line.split(",")
    output = np.array(map(float,input[7:]))
    input = np.array(map(float,input[:7]))
    net.backpropogate(input,output)
    break    