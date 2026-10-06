import numpy as np
import scipy


class NeuralNetwork:

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate

        # Initialize weights between input and hidden layers
        self.wih = np.random.normal(
            0.0,
            pow(self.hnodes, -0.5),
            (self.hnodes, self.inodes)
        )

        # Initialize weights between hidden and output layers
        self.who = np.random.normal(
            0.0,
            pow(self.onodes, -0.5),
            (self.onodes, self.hnodes)
        )

        # Sigmoid activation function
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        # Convert input and target lists into 2D column arrays
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        # Calculate signals into the hidden layer
        hidden_inputs = np.dot(self.wih, inputs)

        # Calculate signals emerging from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals into the final output layer
        final_inputs = np.dot(self.who, hidden_outputs)

        # Calculate signals emerging from the final output layer
        final_outputs = self.activation_function(final_inputs)

        # Calculate output error
        output_errors = targets - final_outputs

        # Calculate hidden layer error
        hidden_errors = np.dot(self.who.T, output_errors)

        # Update weights between hidden and output layers
        self.who += self.lr * np.dot(
            (output_errors * final_outputs * (1.0 - final_outputs)),
            np.transpose(hidden_outputs)
        )

        # Update weights between input and hidden layers
        self.wih += self.lr * np.dot(
            (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
            np.transpose(inputs)
        )

    def query(self, inputs_list):
        # Convert input list into a 2D column array
        inputs = np.array(inputs_list, ndmin=2).T

        # Calculate signals into the hidden layer
        hidden_inputs = np.dot(self.wih, inputs)

        # Calculate signals emerging from the hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals into the final output layer
        final_inputs = np.dot(self.who, hidden_outputs)

        # Calculate signals emerging from the final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs