# MNIST Digit Recognizer
### Digit Recognizer for CSC475 at Louisiana Tech University. It utilizes the MNIST training data set and testing set. Written by Marco Flores.
### Notes
- #### Training set must be named mnist_train.csv
- #### Testing set must be named mnist_test.csv
- #### Saving to a file option produces a .nn file. This is a specially formatted .csv file using the following formatting.
    - First line: Layer sizes separated by commas for each layer
    - Second line and beyond: weights associated with each neuron separated by commas. One neuron per line.
    - The number of neurons is derived from the layer size.
    - Following lines are bias values separated by commas. One layer per line.

### File Tree
- #### build.sh - Shell script to build all the necessary class files
- #### Main.java - Source file for the main interface for the program
- #### weights_and_biases.nn - Representation of a trained network
- #### network - Package containing neural network source files
    - NeuralNetwork.java - Source file for NeuralNetwork class, contains all funcions of the network
    - Neuron.java - Source file and Java class representation of a Neuron