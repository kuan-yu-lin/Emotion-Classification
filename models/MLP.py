'''
Author: Tzu-Ju Lin
'''
# import the library used in our implemetation
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score


class MLP:

    def __init__(self, hidden_layer=(150, 100, 50), max_iter=300, activation='relu'):
        self.hidden_layer_size = hidden_layer
        self.max_iter = max_iter
        self.activation = 'relu'
        self.solver = 'adam'
        self.classifier = MLPClassifier(hidden_layer_sizes=self.hidden_layer_size, max_iter=self.max_iter,
                                        activation=self.activation, solver=self.solver)
        self.pred = None

    def fit(self, X_train, y_train):
        self.classifier.fit(X_train, y_train)

    def test(self, X_test):
        self.pred = self.classifier.predict(X_test)

    def evaluation(self, y_test):
        f1 = f1_score(self.y_pred, y_test, average="macro")
        class_report = classification_report(self.y_pred, y_test, target_names=[
            'joy', "fear", "shame", "sadness", "disgust", "guilt", "anger"])
        print("The f1-score is:")
        print(f1)
        print("The classification report is:")
        print(class_report)


def demo():
    model = MLP()

    # store the  data path
    '''
    The ISEAR dataset we used can be found with the link:
    https://www.unige.ch/cisa/research/materials-and-online-research/research-material/
    After downloading the files, change the value of train_data_file_name to the name of the training data. Change the value of test_data_file_name to the name of the testing data.
    '''
    train_data_path = None
    test_data_path = None

    # read the training data
    df_train = pd.read_csv(train_data_path, delimiter=",", header=None)
    y_train = np.array(df_train.iloc[:, 0])
    X_train = np.array(df_train.iloc[:, 1:])

    # read the testing data
    df_test = pd.read_csv(test_data_path, delimiter=",", header=None)
    y_test = np.array(df_test[0])
    X_test = np.array(df_test.iloc[:, 1:])

    model.fit(X_train, y_train)
    model.test(X_test)
    model.evaluation(y_test)


demo()
