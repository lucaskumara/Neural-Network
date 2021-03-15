import sys
import csv
import math
import random

class Network:

    def __init__(self, learning_rate, inputs, hidden, outputs, examples):
        self.learning_rate = learning_rate  # Learning rate from user input
        self.input_nodes = inputs           # Number of input nodes
        self.hidden_layer_nodes = hidden    # Number of hidden layer nodes
        self.output_nodes = outputs         # Number of output nodes
        self.examples = examples            # .csv filename for training examples

        self.weights = self.network_weights(inputs, hidden, outputs)

    def network_weights(self, inputs, hidden, outputs):
        '''Generates random weights between each layer based on their sizes.'''
        weights = list()

        # Randomly generate weights
        hidden_weights = [[random.choice([-1, 1]) * random.random() for _ in range(inputs)] for _ in range(hidden)]
        output_weights = [[random.choice([-1, 1]) * random.random() for _ in range(hidden)] for _ in range(outputs)]

        weights.append(hidden_weights)
        weights.append(output_weights)

        return weights

    def activation_function(self, value):
        '''Returns the output of the sigmoid activation function given a value.'''
        return 1 / (1 + math.exp(-value))

    def derivative_activation_function(self, value):
        '''Returns the derivative of the activation function given a value.'''
        return self.activation_function(value) * (1 - self.activation_function(value))

    def calculate_inputs(self, previous_layer, weights):
        '''Returns a list of input values for the next layer.'''
        inputs = [0 for _ in range(len(weights))]


        for i in range(len(weights)):
            node_weights = weights[i]

            for j in range(len(previous_layer)):
                inputs[i] += previous_layer[j] * node_weights[j]

        return inputs

    def calculate_outputs(self, inputs):
        '''Returns a list of output values for the current layer.'''
        return [self.activation_function(value) for value in inputs]

    def derivative_input(self, deriv_outputs, inputs):
        '''Calculates the derivative of the error with respect to the inputs.'''
        return [deriv_outputs[i] * self.derivative_activation_function(inputs[i]) for i in range(len(deriv_outputs))]

    def output_derivative_output(self, outputs, expected):
        '''Calculates the derivative of the error with respect to the outputs ONLY FOR OUTPUT LAYER.'''
        return [outputs[i] - expected[i] for i in range(len(outputs))]

    def derivative_output(self, deriv_inputs, weights):
        '''Calculates the derivative of the error with respect to the outputs FOR NON OUTPUT LAYERS.'''
        deriv_outputs = [0 for _ in range(self.hidden_layer_nodes)]

        for i in range(self.hidden_layer_nodes):
            for j in range(len(deriv_inputs)):
                deriv_outputs[i] += deriv_inputs[j] * weights[j][i]

        return deriv_outputs

    def update_weights(self, deriv_inputs, outputs, weights):
        '''Given the outputs from a layer, update the weights between it and the layer ahead.'''
        for i in range(len(deriv_inputs)):
            for j in range(len(outputs)):
                weights[i][j] -= (deriv_inputs[i] * outputs[j] * self.learning_rate)

    def mean_squared_error(self, outputs, expected, example_count):
        '''Returns the mean squared error.'''
        error = 0

        for i in range(example_count):
            for j in range(len(outputs)):
                error += (expected[j] - outputs[j]) / 2

        return error / example_count * len(outputs)

    def train(self, display=False):
        '''Trains the network on all training examples. (1 epoch)'''
        
        # Open training examples
        with open(self.examples, 'r') as file:
            examples = list(csv.reader(file))[:]

            for row in examples:

                float_values = [float(value) for value in row]

                # Identify inputs and expected output
                inputs = float_values[:-self.output_nodes]
                expected_outputs = float_values[-self.output_nodes:]

                ### Determine inputs/outputs

                # Calculate hidden layer inputs and outputs
                hidden_layer_inputs = self.calculate_inputs(inputs, self.weights[0])
                hidden_layer_outputs = self.calculate_outputs(hidden_layer_inputs)

                # Calculate output layer inputs and outputs
                output_inputs = self.calculate_inputs(hidden_layer_outputs, self.weights[1])
                output_outputs = self.calculate_outputs(output_inputs)

                ### Begin backpropogation

                # Get derivative of output layer with respect to inputs/outputs
                output_derivative_o = self.output_derivative_output(output_outputs, expected_outputs)
                output_derivative_i = self.derivative_input(output_derivative_o, output_inputs)

                # Get derivative of hidden layer with respect to inputs/outputs
                hidden_derivative_o = self.derivative_output(output_derivative_i, self.weights[1])
                hidden_derivative_i = self.derivative_input(hidden_derivative_o, hidden_layer_inputs)

                ### Update weights

                # Update weights between each pair of layers
                self.update_weights(hidden_derivative_i, inputs, self.weights[0])
                self.update_weights(output_derivative_i, hidden_layer_outputs, self.weights[1])

                if display:
                    print(f'Input(s): {inputs}, Expected Output(s): {expected_outputs}, Network Output(s): {output_outputs}')

            # Return mean squared error
            return self.mean_squared_error(output_outputs, expected_outputs, len(examples))

if __name__ == '__main__':

    # Get learning rate
    learning_rate = float(sys.argv[1])

    # Get number of nodes
    input_nodes = int(sys.argv[2])
    hidden_layer_nodes = int(sys.argv[3])
    output_nodes = int(sys.argv[4])

    # Get file with training examples
    examples = sys.argv[5]

    # Initialize network
    network = Network(learning_rate, input_nodes, hidden_layer_nodes, output_nodes, examples)

    # Loop for a certain amount of epochs
    epochs = 10000
    for epoch in range(1, epochs + 1):
        error = network.train()
        
        if epoch % 200 == 0:
            print(f'Epoch {epoch}:', error)

    # Output final pieces of information
    print()
    print('Training complete.')
    print()
    print('Hidden Nodes:', network.hidden_layer_nodes)
    print('Learning Rate:', network.learning_rate)
    print('Final MSE: ', error)
    print()
    
    network.train(display=True)
