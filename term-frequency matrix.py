from fnmatch import translate
from string import punctuation
import numpy as np
import pandas as pd


def load_train_dataset():
    import string

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

    # y, X = zip(*rows)
    # X = [x.translate(str.maketrans('', '', string.punctuation)
    #                  ).lower().split() for x in X]


y, X = load_train_dataset()


class BOG(object):

    def __init__(self):
        pass

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
        tfv_df = pd.DataFrame(tfv, columns=unique_word)
        print(tfv_df)
        return tfv_df


b = BOG()
b.extract_word(X)
b.term_freq_matrix(X)
