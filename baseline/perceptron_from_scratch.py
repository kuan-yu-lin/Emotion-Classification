# source:https://www.python-engineer.com/courses/mlfromscratch/06_perceptron/
import numpy as np


class perceptron:

    def __init__(self, learning_rate=0.01, n_iter=100):
        self.weights = None
        self.bias = None
        self.lr = learning_rate
        self.n_iter = n_iter
        self.activation_func = self._uni_step_func

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        for i in range(self.n_iter):
            for index, value in enumerate(X):
                linear_output = np.dot(value, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)
                # update the weight matix and bias
                update = self.lr * (y[index] - y_predicted)
                self.weights += update * value
                self.bias += update

    # predict function
    def predict(self, X):
        # get the dot product of the input X and the weight matrix
        linear_output = np.dot(X, self.weights) + self.bias
        # pass the linear output into the activation function to get y prediction
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    # activation function
    def _uni_step_func(self, x):
        # apply activation function on vectors
        # if x >= 0  return 1, else 0
        return np.where(x >= 0, 1, 0)
