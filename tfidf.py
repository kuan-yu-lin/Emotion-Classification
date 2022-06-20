import numpy as np

class tfidf(object):

    def __init__(self, emotion='joy'):
        self.emotion = emotion
        # list of unique words in the docs
        self.unique_word = None

    def reset_target(self, y):
        reset_y = [1 if i == self.emotion else 0 for i in y]
        return reset_y

    # extract word to get the vocabulary of the training data
    def extract_word(self, X):
        unique_word = []
        for text in X:
            for token in text:
                # remove any number in the text
                if token.isnumeric() == False:
                    if token not in unique_word:
                        unique_word.append(token)

        self.unique_word = unique_word

    # create tf matrix, will later serve as input for the classifier
    def tfidf_matrix(self, X):
        # get the dimension of the tfidf matrix
        M = len(X)
        V = len(self.unique_word)
        # initialize a zero matrix
        tfv = np.zeros((M, V))
        # enumerate through the text
        for i, doc in enumerate(X):
            for word in doc:
                # update the counte of each word in the current text
                if word in self.unique_word:
                    pos = self.unique_word.index(word)
                    tfv[i][pos] += 1

        # create document-frequency array
        dfa = np.count_nonzero(tfv > 0, axis=0)
        # initialize a zero array for inversed document-frequency array
        idf = np.zeros(np.shape(dfa))
        # create inversed document-frequency array
        for i, d in enumerate(dfa):
            idf[i] = np.log(len(X)/(d + 1))
        
        r, c = tfv.shape
        # initialize a zero matrix
        tfidf = np.zeros(np.shape(tfv))
        # get the tfidf matrix with the equation: idf = log(N/df + 1)
        for x in range(r):
            for y in range(c):
                tfidf[(x, y)] = tfv[x][y] * idf[y]

        return tfidf
