#@Leester Mei
import math
import numpy as np
class NeuralNetwork:
  def __init__(self,sizes,learning_rate,error_threshhold):
    self.layers = len(sizes)
    self.sizes = sizes
    self.learning_rate = learning_rate
    self.error_threshhold = error_threshhold
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
        for k in range(sizes[i+1]):
          w.append(np.random.randn())
      self.weights.append(w)
  
  #Calculate error using mean squared error function
  def calculate_error(self,output,target):
    return (0.5)*(pow((output-target),2))
  
  #Propogate input data foward by one layer, starting with input data from layer ending in output data in layer+1   
  def forwardpropogate(self,input,layer):
    output = []
    for i in range(self.sizes[layer+1]):
      o = 0
      for j in range(self.sizes[layer]):
        o += input[j]*self.weights[layer][j+(self.sizes[layer]*i)]
      output.append(self.sigmoid(o+self.biases[layer][i]))
    return output
  
  #Back propogation algorithm to adjust weights and biases based on target values and neural network outputs
  def backpropogate(self,input,output):
    O_j = self.forwardpropogate(input,0) #Output of hidden layer
    O_k = self.forwardpropogate(O_j,1) #Output of the output layer
    #Compute Errors
    delta_k = [] #Output layer Error
    for i in range(self.sizes[2]): #For each output layer node output, compute its error
      delta_k.append(O_k[i]*(1-O_k[i])*(O_k[i] - output[i]))
    delta_j = [] #Hidden Layer Error
    for i in range(self.sizes[1]): #For each hidden layer node output, compute its error
      sum = 0
      for j in range(self.sizes[2]): #For each output layer node, compute the product of the output error, delta_k, and weight from node in layer j to layer k
        sum += delta_k[j]*self.weights[1][i]
      delta_j.append(O_j[i]*(1-O_j[i])*(sum))
    #Begin backpropogation algorithm
    #Adjust Weights
    for w in range(len(self.weights)):
      for l in range(len(self.weights[w])):
        if w == 0: #Hard code cases since deltas are in seperate arrays, better implementation would've been 2-d Array
          self.weights[w][l] = self.weights[w][l]-(self.learning_rate)*delta_j[int(l/self.sizes[w])]*input[l%self.sizes[w]]
        elif w == 1:
          self.weights[w][l] = self.weights[w][l]-(self.learning_rate)*delta_k[int(l/self.sizes[w])]*O_j[l%self.sizes[w]]
    #Adjusts Biases
    for b in range(len(self.biases)):
      for l in range(len(self.biases[b])):
        if b == 0:
          self.biases[b][l] = self.biases[b][l]-(self.learning_rate)*(delta_j)[l]
        elif b == 1:
          self.biases[b][l] = self.biases[b][l]-(self.learning_rate)*(delta_k)[l]
    #End backpropogation algorithm
  
  def train(self,training_data):
    below_Error_Thresh = False
    counter = 0
    while not below_Error_Thresh:
      below_Error_Thresh = True
      for i in range(len(training_data)):
        counter+=1
        inp = training_data[i][:7]
        out = training_data[i][7:]
        self.backpropogate(inp,out)
        if self.validate(training_data[i]) >= self.error_threshhold:
          below_Error_Thresh = False
    return "Trained {} because of error on a fold".format(counter)    
    
  #Validate accuracy of model given validation set
  def validate(self,data):
    validation_set = data[:7]
    target_set = data[7:]
    output = self.forwardpropogate(validation_set,0)
    output = self.forwardpropogate(output,1)
    return abs(output[0]-target_set[0])
    #return self.calculate_error(output[0],target_set[0])
    
  @staticmethod
  #Our squashing/logistic/activation function given input vector z
  def sigmoid(z):
    return 1/(1+np.exp(-z))
    
data = []  
import sys
with open(sys.argv[1],'r') as f: #Import normalized data
  for line in f: #Each line in data is 7 input values, and 1 target output value, delimited by commas
    input = line.split(",")
    data.append(np.array(map(float,input)))
#N-Fold Cross Validation
  net_out = open("neural_network_parameters.txt","w")
  n = 5
  folds = []
  for f in range(n):
    folds.append([])
  for i in range(len(data)):  #Create Folds
    for f in range(n):
      if int(i/42) != f:
        folds[f].append(data[i])
  #Begin Cross Validation
  for i in range(n):
    net_out.write("Epoch {}\n".format(i))
    net = NeuralNetwork([7,3,1],0.5,0.1)
    for j in range(n):  #Train on all n-1 folds
      if i != j:
        net_out.write("{}".format(net.train(folds[j])))
    #net.validate(folds[i]) #Validate on 1 fold
    for k in range(len(net.weights)):
      net_out.write("Weights: \n")
      for k2 in range(len(net.weights[k])):
        net_out.write("{} ".format(net.weights[k][k2]))
      net_out.write("\n")
    for b in range(len(net.biases)):
      net_out.write("Biases: \n")
      for b2 in range(len(net.biases[b])):
        net_out.write("{} ".format(net.biases[b][b2]))
      net_out.write("\n")
    net_out.write("____________________________________\n")
    