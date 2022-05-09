from fnmatch import translate
from random import randint
from string import punctuation
import string
import numpy as np
import pandas as pd
from perceptron_from_scratch import perceptron
from evaluation import confusion_matrix, precision, recall, f1score


def load_train_dataset():

    with open('isear-train.txt', mode='r', encoding='utf-8') as f:
        rows = [l.strip().split('\t')[:2] for l in f]
    y = []
    X = []
    for row in rows:
        if len(row) == 2:
            y.append(row[0])
            X.append(row[1].translate(str.maketrans(
                '', '', string.punctuation)).lower().split())
    return y, X


y, X = load_train_dataset()


class BOG(object):

    def __init__(self, emotion='joy'):
        self.emotion = emotion

    def reset_target(self, y):
        reset_y = [1 if i == self.emotion else 0 for i in y]
        return reset_y

    def extract_word(self, X):
        unique_word = []
        word_dict = {}
        for text in X:
            for token in text:
                if token.isnumeric() == False:
                    if token not in unique_word:
                        unique_word.append(token)

        return unique_word, word_dict

    def term_freq_matrix(self, X):
        unique_word, word_dict = self.extract_word(X)
        # create term-frequency matrix
        M = len(X)
        V = len(unique_word)
        tfv = np.zeros((M, V))
        for i, doc in enumerate(X):
            for word in doc:
                if word in unique_word:
                    pos = unique_word.index(word)
                    tfv[i][pos] += 1

        return tfv


b = BOG()
tfv = b.term_freq_matrix(X)
reset_y = b.reset_target(y)


p = perceptron()
p.fit(tfv[:2000], y[:2000])
y_pred = p.predict(tfv[2000:2400])


m = confusion_matrix(y_pred, reset_y[2000:2400])
fscore = f1score(y_pred, reset_y[2000:2400])
print(m, fscore)
