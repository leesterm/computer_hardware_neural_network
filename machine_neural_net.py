#@Leester Mei
import math
import numpy as np
class NeuralNetwork:
  def __init__(self,sizes,learning_rate,error_threshold):
    self.layers = len(sizes)
    self.sizes = sizes
    self.learning_rate = learning_rate
    self.error_threshold = error_threshold
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
    for i in range(len(training_data)):
      self.backpropogate(training_data[i][:7],training_data[i][7:])
        
  #Validate accuracy of current model given validation set
  def validate(self,data,epoch,out):
    total_error = 0
    for i in range(len(data)):
      validation = data[i][:7]
      target_data = data[i][7:]      
      output = self.forwardpropogate(validation,0)
      output = self.forwardpropogate(output,1)
      total_error += self.calculate_error(output[0],target_data[0])
    out.write("{}|{}\n".format(epoch,total_error))
    if total_error > self.error_threshold:
      return False
    else:
      return True
    
  #After training and validation, provide unseen testing date as a final evaluation of accuracy
  def test(self,data,output):
    output.write("Testing\n")
    output.write("Output | Target | MSE\n")
    total_error = 0
    for i in range(len(data)):
      test_data = data[i][:7]
      target_data = data[i][7:]
      test_output = self.forwardpropogate(test_data,0)
      test_output = self.forwardpropogate(test_output,1)
      error = self.calculate_error(test_output[0],target_data[0])
      total_error += error
      output.write("{} {} {}\n".format(test_output[0],target_data[0],error))
    output.write("Total MSE: {}\n".format(total_error))
    output.write("Average MSE: {}".format(total_error/len(data)))
    
  @staticmethod
  #Our squashing/logistic/activation function given input z
  def sigmoid(z):
    return 1/(1+np.exp(-z))
    
data = []  
import sys
with open(sys.argv[1],'r') as f: #Import normalized data
  for line in f: #Each line in data is 7 input values, and 1 target output value, delimited by commas
    input = line.split(",")
    data.append(np.array(map(float,input)))
  err_thresh = float(sys.argv[2])
  net = NeuralNetwork([7,3,1],0.5,err_thresh)
  #N-Fold Cross Validation
  net_out = open("{}_nn_parameters.txt".format(err_thresh),"w")
  training_out = open("{}_training_error.txt".format(err_thresh),"w")
  testing_out = open("{}_testing.txt".format(err_thresh),"w")
  n = 5
  testing = []
  training_validation = []
  for i in range(len(data)):
    if i < len(data)/n:
      testing.append(data[i])
    else:
      training_validation.append(data[i])
  #Begin training and validating
  epoch = 0
  while True:
    #First randomly reorder the training and validation array in order to avoid over training over training due to the same training sets
    for i in range(len(training_validation)):
      training_validation[i] = training_validation[np.random.randint(0,len(training_validation)-1)]
    net.train((training_validation[:126]))#Hard code 126, Better implementation would be length of training validation set - length of training validation set / n-1
    below_error_thresh = net.validate(training_validation[126:],epoch,training_out)
    epoch += 1
    if below_error_thresh == True:
      break
  #Begin testing
  net.test(testing,testing_out)
  #Output our weights and biases for our trained neural network
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
    
  net_out.close()
  training_out.close()
  testing_out.close()