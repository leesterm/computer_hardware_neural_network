Program was tested and ran using Python 2.7.12
Dependencies: numpy
Data from: http://archive.ics.uci.edu/ml/machine-learning-databases/cpu-performance/machine.data
There are 3 programs:
  -machine_neural_net.py
    The main program which includes the NeuralNetwork class and implementation of back propogation
    To Run:
      python machine_neural_net.py machine_data_normalized.txt [error_threshold]
    After running the program, it will produce three .txt files:
      -[error_threshold]_training_error:
        This file contains the total output error during training and validation
      -[error_threshold]_testing_error:
        This file contains the testing results
      -[error_threshold]_nn_parameters:
        This file contains the weights and biases
  -normalize_machine_data.py
    This program was used to normalize the data
    To Run:
      python normalize_machine_data.py machine_data.txt
    After running the program it will produce a .txt file with the normalized data.
  -error_plot.py
    This program was used to plot the total output error for each iteration in training and validation