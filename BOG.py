import numpy as np


class BOG(object):

    def __init__(self, emotion='joy'):
        self.emotion = emotion
        # list of unique words in the docs
        self.unique_word = None
        # dictionary that keeps track of word frequenciess
        self.word_dict = None

    def reset_target(self, y):
        reset_y = [1 if i == self.emotion else 0 for i in y]
        return reset_y

    # extract word to get the vocabulary of the training data
    def extract_word(self, X):
        unique_word = []
        word_dict = {}
        for text in X:
            for token in text:
                # remove any number in the text
                if token.isnumeric() == False:
                    if token not in unique_word:
                        unique_word.append(token)

        self.unique_word = unique_word
        self.word_dict = word_dict

    # create tf matrix, will later serve as input for the classifier
    def term_freq_matrix(self, X):
        # get the dimension of the term-frequency matrix
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
        # just to visualize
        # df_tfd = pd.DataFrame(tfv, columns=unique_word)
        return tfv
