# Neural Network From Scratch

A feed-forward neural network for MNIST handwritten digit classification, implemented manually with NumPy and SciPy as a learning project during the RUDN Summer School.

---

## About the Project

This repository contains a small neural network implemented from scratch, created as part of my learning experience during the **RUDN Summer School**, where I studied the fundamentals of neural networks.

The network logic (weight initialization, forward propagation, error calculation, backpropagation and weight updates) is written by hand instead of relying on high-level deep-learning frameworks such as TensorFlow or PyTorch.

This does not mean that every numerical operation is implemented from scratch. The project uses **NumPy** for array and matrix operations, **SciPy** for the sigmoid function (`scipy.special.expit`) and image rotation (`scipy.ndimage.rotate`), and **Matplotlib** to display a test image.

The goal of the project is to understand the basic mechanics of a neural network by implementing them directly.

---

## Neural Network Architecture

The network in `main.py` has one hidden layer and uses the following configuration:

- 784 input nodes
- 200 hidden nodes
- 10 output nodes
- Sigmoid activation function
- Learning rate of 0.1

```text
Input Layer
784 neurons
     │
     ▼
Hidden Layer
200 neurons
     │
     ▼
Output Layer
10 neurons
     │
     ▼
Digit Prediction (0–9)
```

**Why 784 input nodes?** Each MNIST image is 28 × 28 pixels, which gives 28 × 28 = 784 pixel values. Each pixel is fed to one input node.

**Why 10 output nodes?** MNIST has ten digit classes (0–9). Each output node corresponds to one digit, and the predicted digit is the index of the output node with the highest activation.

---

## How the Neural Network Works

The implementation is in `NeuralnetworksFromScratch.py`, in the `NeuralNetwork` class, which provides an `__init__` method, a `train` method and a `query` method.

### Weight Initialization

Two weight matrices are created with `np.random.normal`, using a mean of `0.0`:

- `wih`: weights between the input and hidden layers, with shape `(hidden_nodes, input_nodes)` and standard deviation `hidden_nodes ** -0.5`
- `who`: weights between the hidden and output layers, with shape `(output_nodes, hidden_nodes)` and standard deviation `output_nodes ** -0.5`

There are no bias terms in this implementation.

### Forward Propagation

Both `train` and `query` perform the same forward pass:

```text
Input
  ↓
Input → Hidden weighted sum        (wih · inputs)
  ↓
Sigmoid activation
  ↓
Hidden → Output weighted sum       (who · hidden_outputs)
  ↓
Sigmoid activation
  ↓
Final output
```

The sigmoid function is computed with `scipy.special.expit`.

### Error Calculation

The output error is the difference between the target and the actual output:

```text
Output Error = Target - Final Output
```

### Backpropagation

The error is propagated back to the hidden layer by multiplying the transposed hidden-to-output weights with the output errors:

```text
Hidden Error = who.T · Output Error
```

Each hidden node therefore receives a share of the output error according to the weights connecting it to the output nodes. In this implementation the result is not normalized.

### Weight Updates

Each weight matrix is adjusted using the learning rate, the error of the layer it feeds into, the derivative of the sigmoid function, and the output of the preceding layer. Using `o` for a layer's sigmoid output, the sigmoid derivative is expressed as `o * (1 - o)`.

```text
Δwho = learning_rate · (output_errors · final_outputs · (1 - final_outputs)) · hidden_outputsᵀ
Δwih = learning_rate · (hidden_errors · hidden_outputs · (1 - hidden_outputs)) · inputsᵀ
```

Both updates are applied after every single training example. There is no mini-batching.

---

## Input Normalization and Target Encoding

Both are implemented in `main.py`.

### Input Normalization

Raw MNIST pixel values are integers from 0 to 255. They are rescaled with:

```python
(all_values[1:] / 255.0 * 0.99) + 0.01
```

A pixel value of 0 becomes `0.01` and a value of 255 becomes `1.0`, so inputs fall approximately in the range `0.01–1.0`. Inputs are scaled to a small range so that they suit the sigmoid activation, and the small offset avoids exact zero inputs. A zero input would produce no weight update for its connections, since the update is proportional to the input value.

### Target Encoding

The label is converted into a vector of 10 target values:

```python
targets = np.zeros(output_nodes) + 0.01
targets[int(all_values[0])] = 0.99
```

The output node of the correct digit receives the target `0.99`, and the other nine output nodes receive `0.01`. The sigmoid function never produces exactly 0 or 1, so these values are targets the outputs can actually approach.

---

## Data Augmentation

`main.py` applies a simple rotation-based augmentation. For every training record, the network is trained on three versions of the same image:

1. The original image
2. The image rotated by **+10 degrees**
3. The image rotated by **−10 degrees**

The rotation uses `scipy.ndimage.rotate` with `reshape=False` (the image stays 28 × 28) and `cval=0.01`, so pixels that fall outside the original image are filled with the same value that represents the background after normalization. All three versions use the same target vector.

Handwritten digits vary naturally in slant and orientation, so small rotations expose the network to some of that variation. This project does not include an experiment comparing training with and without augmentation, so no claim is made about its effect on accuracy.

---

## Dataset

The project uses the **MNIST handwritten digit dataset**:

- 28 × 28 grayscale images
- 784 pixel values per image
- 10 digit classes (0–9)

The data is stored as CSV files. Each line contains the label followed by the 784 pixel values.

`LibModule.py` downloads the dataset automatically when the required CSV files are not found locally. The download and extraction happen only if `datasets/mnist_train.csv` or `datasets/mnist_test.csv` is missing, so an internet connection is needed on the first run. The files are downloaded from a GitHub-hosted copy of MNIST, as specified by the URLs in `LibModule.py`.

The dataset is intentionally excluded from Git version control because of its size. The `datasets/` directory is created and used locally.

---

## Project Structure

```text
RUDN-Neural-Networks-From-Scratch/
│
├── NeuralnetworksFromScratch.py
├── main.py
├── LibModule.py
├── requirements.txt
├── README.md
└── .gitignore
```

| File | Purpose |
|---|---|
| `NeuralnetworksFromScratch.py` | Defines the `NeuralNetwork` class: weight initialization, `train` (forward pass, backpropagation, weight updates) and `query` (forward pass only). |
| `main.py` | Entry point. Creates the network, loads the data, trains with rotation augmentation, shows and classifies one test example, and evaluates on the full test set. |
| `LibModule.py` | Provides `load_mnist()`, which downloads and extracts the MNIST CSV files if needed and returns the training and test records as lists of lines. |

---

## Requirements

```text
numpy
scipy
matplotlib
```

Modules such as `pathlib`, `urllib` and `zipfile` are part of the Python standard library and do not need to be installed.

---

## Installation and Usage

```bash
git clone <repository-url>
cd <repository-name>
pip install -r requirements.txt
python main.py
```

On the first run, the MNIST dataset is downloaded automatically into the `datasets/` directory.

What `main.py` does when run:

1. Trains the network for 5 epochs on the training data (with augmentation).
2. Prints the label of the first test record and displays its image with Matplotlib. Close the image window to continue.
3. Prints the network output for that example.
4. Evaluates the network on the whole test set and prints the number of correct answers and the performance percentage.

Training processes every training record three times per epoch (original and two rotations), and the weights are updated one example at a time, so it can take a while to finish.

---

## Training Configuration

| Parameter | Value |
|---|---:|
| Input Nodes | 784 |
| Hidden Nodes | 200 |
| Output Nodes | 10 |
| Learning Rate | 0.1 |
| Epochs | 5 |
| Activation Function | Sigmoid |
| Data Augmentation | ±10° rotation |

---

## Results

The script evaluates the trained network on the MNIST test dataset and reports the number of correct predictions and the overall classification accuracy. For each test image, the predicted digit is the output node with the highest activation (`np.argmax`).

No accuracy value is recorded in this repository. Results will vary slightly between runs because the weights are initialized randomly.

---

## Learning Outcomes

Implementing this project helped in understanding:

- Neural network architecture (input, hidden and output layers)
- Weight initialization with a normal distribution
- Forward propagation
- The sigmoid activation function
- Error calculation
- Backpropagation of errors to the hidden layer
- Weight updates using the learning rate and sigmoid derivative
- Input normalization and target encoding
- Data augmentation through small rotations
- MNIST digit classification

---

## Educational Context

This implementation was created as part of my learning experience during the **RUDN Summer School**. It is an educational project intended to practice the fundamentals of neural networks, not a production-ready library.

---

## License

No license has been specified for this project at the moment.