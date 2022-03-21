#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
# import imageio as iio
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

# Load the datasets
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
# #Test the images and their respective labels
# index = 1
# plt.imshow(train_set_x_orig[index])
# print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
# plt.show()

# Create variables of dataset sizes
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

# Reshape datasets into vectors
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# Normalize the data
train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

# Variables and their sizes/values
print("Matrices:\n")
print("train_set_x_orig: " + str(train_set_x_orig.shape) + "\n")
print("train_set_x: " + str(train_set_x.shape) + "\n")
print("train_set_x_flatten: " + str(train_set_x_flatten.shape) + "\n")
print("train_set_y: " + str(train_set_y.shape) + "\n")
print("test_set_x_orig: " + str(test_set_x_orig.shape) + "\n")
print("test_set_x: " + str(test_set_x.shape) + "\n")
print("test_set_x_flatten: " + str(test_set_x_flatten.shape) + "\n")
print("test_set_y: " + str(test_set_y.shape) + "\n")
print("\n")
print("Variables:\n")
print("m_train: " + str(m_train) + "\n")
print("m_test: " + str(m_test) + "\n")
print("num_px: " + str(num_px) + "\n")
print("\n")

# Function to calculate the sigmoid function
def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    s = 1/(1+np.exp(-z))
    return s
# #Test the sigmoid function
# print("sigmoid([0,2]) = " + str(sigmoid(np.array([0,2]))))

# Function to initialize parameters
def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """

    w = np.zeros((dim,1))
    b = 0

    assert(w.shape == (dim,1))
    assert(isinstance(b, float) or isinstance(b, int))

    return w,b
# # Test the initialize_with_zeros function
# dim = 2
# w,b = initialize_with_zeros(dim)
# print("w = " + str(w))
# print("b = " + str(b))

# Function for forward and backward propagation
def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b

    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """

    m = X.shape[1]

    # Forward Propagation (X to Cost)
    A = sigmoid(w.T @ X + b) #1x209
    cost = -1/m * ((Y @ (np.log(A).T)) + ((1-Y) @ (np.log(1-A).T)))

    # Backward Propagation (find Grad)
    dw = 1/m * X @ ((A-Y).T)
    db = 1/m * sum(sum(A - Y))

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {"dw":dw,
             "db":db}

    return grads, cost
# # Test the propagate function - initialize w, b, X, Y. Calculate grads (dw and db) and cost.
# w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
# grads, cost = propagate(w, b, X, Y)
# print("dw = " + str(grads["dw"]))
# print("db = " + str(grads["db"]))
# print("cost = " + str(cost))

# Function for optimization
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost):
    """
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """

    costs = []

    for i in range(num_iterations):
        # Cost and gradient calculation
        grads, cost = propagate(w, b, X, Y)

        # Derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # Update rule
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print every 100th cost
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" %(i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs
# # Test the optimize function - optimize w, b, dw, db.
# params, grads, costs = optimize(w, b, X, Y, num_iterations=100, learning_rate = 0.009, print_cost=False)
# print("w = " + str(params["w"]))
# print("b = " + str(params["b"]))
# print("dw = " + str(grads["dw"]))
# print("db = " + str(grads["db"]))

# Function for prediction
def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''

    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)

    # Compute vector "A", predicting the possibility for a cat
    A = sigmoid(w.T @ X + b)

    Y_prediction = np.rint(A).astype(int) # Possibly remove the .astype(int)

    assert(Y_prediction.shape == (1,m))

    return Y_prediction
# # Test the predict function - predict the outputs
# w = np.array([[0.1124579],[0.23106775]])
# b = -0.3
# X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
# print("predictions = " + str(predict(w, b, X)))

# Function to merge all functions into a model
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously

    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations

    Returns:
    d -- dictionary containing information about the model.
    """

    # Parameter initialization
    w, b = initialize_with_zeros(X_test.shape[0])

    # Gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # Retrieve parameters w and b from dictionary 'parameters'
    w = parameters["w"]
    b = parameters["b"]

    # Predict the test/train examples
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    # Setting the dictionary
    d = {"costs": costs,
        "Y_prediction_train": Y_prediction_test,
        "Y_prediction_test": Y_prediction_test,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations}

    return d

# Running the model
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)

# # Example of a picture that was wrongly classified.
# index = 1
# plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
# print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + classes[d["Y_prediction_test"][0,index]].decode("utf-8") +  "\" picture.")

# Plot of the learning curve with costs
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

# Variation of the learning rate to show the effect thereof on the classification accuracy and training speed
learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')
for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))
plt.ylabel('cost')
plt.xlabel('iterations (hundreds)')
legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()

# Testing the trained model on individual images for interest's sake
for i in range(17):
    # Loading the image
    image_name = str(i) + ".jpg"
    image_path = "images/" + image_name
    image = Image.open(image_path)
    # Preprocessing the image
    image_arr = np.array(image)
    temp1 = Image.fromarray(image_arr)
    temp2 = np.array(temp1.resize((num_px, num_px)))
    temp3 = temp2.reshape((1, num_px*num_px*3))
    my_image = temp3.T
    # Predict the label
    my_predicted_image = predict(d["w"], d["b"], my_image)
    plt.imshow(image)
    print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
    plt.show()
print("Exiting...")
