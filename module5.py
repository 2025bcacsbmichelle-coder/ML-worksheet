import numpy as np
import pandas as pd

# Define the sigmoid activation function and its derivative function, to use later.
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
  return x * (1 - x)

# Sample Input dataset
X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])

# Sample Output dataset
y = np.array([[0],
              [1],
              [1],
              [0]])

np.random.seed(1)

# Initializing weights randomly, but keeping mean of weights as zero.
w0 = 2 * np.random.random((3,4)) - 1
w1 = 2 * np.random.random((4,1)) - 1

# To collect errors for plotting later
errors = []

# Iterate over 5000 epochs
for i in range(5000):

  # Forward propagation
  a0 = X                              # input layer
  a1 = sigmoid(np.dot(a0, w0))        # hidden layer
  a2 = sigmoid(np.dot(a1, w1))        # output layer

  # Calculate the error in model output (layer2)
  output_error = y - a2

  # Storing the error for every epoch
  errors.append({
    "epochs": i,
    "error": np.mean(np.abs(output_error))
  })

  # Printing the error every 500 steps
  if (i % 500) == 0:
    print("Error after "+str(i)+" epochs:" + str(np.mean(np.abs(output_error))))

  # Back propagation
  layer2_delta = output_error * sigmoid_derivative(a2)
  layer1_error = layer2_delta.dot(w1.T)
  layer1_delta = layer1_error * sigmoid_derivative(a1)

  # Update weights
  w1 -= np.dot(a1.T, layer2_delta)
  w0 -= np.dot(a0.T, layer1_delta)

    # Plotting Graph of errors vs epochs

pd.DataFrame(errors).plot(x='epochs', y='error')
