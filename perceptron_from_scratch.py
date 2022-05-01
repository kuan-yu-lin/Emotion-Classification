#source:https://www.python-engineer.com/courses/mlfromscratch/06_perceptron/
import numpy as np

#input: 特徵vector
#output: dicision boundry?
#目標是找到可以分開兩個class的一個向量

class perceptron:
    
    def __init__(self, learning_rate=0.01, epoch = 30):
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate
        self.epoch = epoch


    #activation function
    def model(self,x):
        if np.dot(self.w, x) >= self.b:
            return 1
        else:
            return 0

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_true = y

        for i in range(self.epoch):
            for index, value in enumerate(X):
                linear_output = np.dot(index, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)

                update = self.lr * (y_true[index] - y_predicted)

                self.weights += update * value
                self.bias += update
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted
    
    def _unit_step_func(self, x):
        return np.where(x>=0, 1, 0)



        
    

