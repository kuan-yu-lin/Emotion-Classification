# This is the version for evaluating tfidf implement.
# We implement tfidf from scratch here.

import numpy as np

class tfidf(object):

    def __init__(self, emotion='joy'):
        self.emotion = emotion
        self.unique_word = None

    def reset_target(self, y):
        reset_y = [1 if i == self.emotion else 0 for i in y]
        return reset_y

    def extract_word(self, X):
        unique_word = []
        for text in X:
            for token in text:
                if token.isnumeric() == False:
                    if token not in unique_word:
                        unique_word.append(token)

        self.unique_word = unique_word

    def tfidf_matrix(self, X):
        M = len(X)
        V = len(self.unique_word)
        tfv = np.zeros((M, V))
        for i, doc in enumerate(X):
            for word in doc:
                if word in self.unique_word:
                    pos = self.unique_word.index(word)
                    tfv[i][pos] += 1

        # create document-frequency array
        dfa = np.count_nonzero(tfv > 0, axis=0)
        idf = np.zeros(np.shape(dfa))

        # create inversed document-frequency array: idf = log(N/df + 1)
        for i, d in enumerate(dfa):
            idf[i] = np.log(len(X)/(d + 1))
        
        r, c = tfv.shape
        tfidf = np.zeros(np.shape(tfv))

        for x in range(r):
            for y in range(c):
                tfidf[(x, y)] = tfv[x][y] * idf[y]


        return tfidf
