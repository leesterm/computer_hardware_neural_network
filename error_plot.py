#@Leester Mei
import numpy as np
import matplotlib.pyplot as plt

data = []
index = []
count = 0
average = 0
import sys 
with open(sys.argv[1],'r') as f:
  plt.subplot(111)
  for line in f:
    if line != "_______________\n":
      data.append(float(line))
      index.append(count)
      average += float(line)
      count+=1
    else:
      print average/count
      plt.scatter(np.array(index),np.array(data))
      plt.xlabel('Iteration')
      plt.ylabel('Total Output Error')
      plt.figure()
      data = []
      index = []
      count = 0
      average = 0
plt.show()
  