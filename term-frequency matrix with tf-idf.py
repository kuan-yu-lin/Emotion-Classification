from fnmatch import translate
from random import randint
from string import punctuation
from numpy import ma
import string
import numpy as np
import pandas as pd
from perceptron_from_scratch import perceptron
from evaluation import confusion_matrix, precision, recall, f1score
from BOG import BOG


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

    def tf_idf_matrix(self, X):
        # create document-frequency array
        tfv = self.term_freq_matrix(X)
        dfa = np.count_nonzero(tfv > 0, axis=0)
        idf = np.zeros(np.shape(dfa))

        # create inversed document-frequency array: idf = log(N/df + 1)
        for i, d in enumerate(dfa):
            idf[i] = np.log(len(X)/(d + 1))

        # create tf-idf matrix: tf-idf = tf * idf
        tfidf = np.zeros(np.shape(tfv))
        for x, y in np.ndindex(tfv.shape):
            tfidf[(x, y)] = tfv[x][y] * idf[y]

        return tfidf


b = BOG()
tfidf = b.tf_idf_matrix(X)
reset_y = b.reset_target(y)


p = perceptron()
p.fit(tfidf[:3000], y[:3000])
y_pred = p.predict(tfidf[3000:3500])


m = confusion_matrix(y_pred, reset_y[3000:3500])
fscore = f1score(y_pred, reset_y[3000:3500])
print(m, fscore)
