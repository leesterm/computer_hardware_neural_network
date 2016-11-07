Program was tested and ran using Python 2.7.12
Dependencies: numpy
There are 3 programs:
  -machine_neural_net.py
    The main program which includes the NeuralNetwork class and implementation of back propogation
    To Run:
      python machine_neural_net.py machine_data_normalized.txt
    After running the program, it will produce two .txt files:
      -neural_network_error_output_---:
        This file contains the total output error during training
      -neural_network_parameters_---:
        This file contains the validation results as well as the weights and biases of the neural network
  -normalize_machine_data.py
    This program was used to normalize the data
    To Run:
      python normalize_machine_data.py machine_data.txt
    After running the program it will produce a .txt file with the normalized data.
  -error_plot.py
    This program was used to plot the total output error for each iteration in training