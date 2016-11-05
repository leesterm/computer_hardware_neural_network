#@Leester Mei
import math
import numpy as np
class NeuralNetwork:
  def __init__(self, sizes):
    self.layers = len(sizes)
    self.sizes = sizes
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
    O_h = net.forwardpropogate(input,0) #Output of hidden layer
    O_o = net.forwardpropogate(O_h,1) #Output of the output layer
    #Begin backpropogation algorithm
    
    #End backpropogation algorithm
    
  @staticmethod
  #Our squashing/logistic/activation function given input vector z
  def sigmoid(z):
    return 1/(1+np.exp(-z))
  
net = NeuralNetwork([7,3,1])
import sys
with open(sys.argv[1],'r') as f: #Import normalized data
  for line in f: #Each line in data is 7 input values, and 1 target output value, delimited by ","
    input = line.split(",")
    output = np.array(map(float,input[7:]))
    input = np.array(map(float,input[:7]))
    net.backpropogate(input,output)
    break    