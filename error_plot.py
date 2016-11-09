#@Leester Mei
import numpy as np
import matplotlib.pyplot as plt

data = []
index = []
count = 0
average = 0
plt.subplot(111)
import sys 
with open(sys.argv[1],'r') as f:
  for line in f:
    d = str.split(line,"|")
    data.append(float(d[1]))
    index.append(count)
    count+=1

plt.scatter(np.array(index),np.array(data))
plt.xlabel('Iteration')
plt.ylabel('Total Output Error')
plt.figure()

training_error = [0.0725080796507,0.0095483356193,0.000161006150204,4.71004049573e-05,9.47749828811e-06,9.60042080888e-07]
testing_error = [0.639960803345,0.464252992502,0.349165200076,0.260834483893,0.334285022316,0.493145938006]
error_threshold = [1,2,3,4,5,6]
plt.plot(np.array(error_threshold),np.array(training_error))
plt.plot(np.array(error_threshold),np.array(testing_error))
plt.xlabel('Error Threshold')
plt.xticks(error_threshold,[0.1,0.01,0.001,0.0001,0.00001,0.000001])
plt.ylabel('Error')
plt.ylim([0,.65])
plt.figure()
'''
predicted = []
target = []
index = []
count = 1
import sys 
with open('1e-05_testing.txt','r') as f:
  for line in f:
    d = str.split(line," ")
    predicted.append(d[0])
    target.append(d[1])
    index.append(count)
    count+=1
plt.plot(np.array(index),np.array(predicted))
plt.plot(np.array(index),np.array(target))
plt.xlabel("Tests")
plt.ylabel("Predicted/Target CPU Performance")
plt.show()
'''  