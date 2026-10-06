from NeuralnetworksFromScratch import NeuralNetwork
from LibModule import load_mnist

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate


# Neural network configuration
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1

network = NeuralNetwork(
    input_nodes,
    hidden_nodes,
    output_nodes,
    learning_rate
)

# Load MNIST training and testing data
train_data_list, test_data_list = load_mnist()

epochs = 5


# Train the neural network
for epoch in range(epochs):
    for record in train_data_list:

        # Convert CSV record into a NumPy array
        all_values = record.split(",")
        all_values = np.array(all_values, dtype=np.uint8)

        # Normalize pixel values to the range 0.01 - 1.0
        inputs = (all_values[1:] / 255.0 * 0.99) + 0.01

        # Create target values
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99

        # Train using the original image
        network.train(inputs, targets)

        # Reshape image for data augmentation
        img_array = inputs.reshape(28, 28)

        # Train using rotated versions of the image
        rotated1 = rotate(
            img_array,
            10,
            cval=0.01,
            reshape=False
        )

        rotated2 = rotate(
            img_array,
            -10,
            cval=0.01,
            reshape=False
        )

        network.train(rotated1.flatten(), targets)
        network.train(rotated2.flatten(), targets)


# Test the network using a single example
record = test_data_list[0]
all_values = record.split(",")
print("Label:", all_values[0])

all_values = np.array(all_values, dtype=np.uint8)

# Display the test image
img_array = all_values[1:].reshape(28, 28)

plt.imshow(
    img_array,
    cmap="Greys",
    interpolation=None
)
plt.show()


# Classify the example using the trained network
example_input = (all_values[1:] / 255.0 * 0.99) + 0.01

print("Network output:")
print(network.query(example_input))


# Evaluate the network on the complete test dataset
scorecard = []

for record in test_data_list:

    all_values = record.split(",")
    all_values = np.array(all_values, dtype=np.uint8)

    correct_label = int(all_values[0])

    # Normalize input values
    inputs = (all_values[1:] / 255.0 * 0.99) + 0.01

    # Query the trained network
    outputs = network.query(inputs)

    # Select the output neuron with the highest activation
    predicted_label = np.argmax(outputs)

    if predicted_label == correct_label:
        scorecard.append(1)
    else:
        scorecard.append(0)


# Calculate and display performance
correct_answers = sum(scorecard)
accuracy = (correct_answers / len(scorecard)) * 100

print("Correct Answers:", correct_answers)
print("Performance:", accuracy, "%")
