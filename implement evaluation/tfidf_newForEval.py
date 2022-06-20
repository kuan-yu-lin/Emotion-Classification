# This is the version for evaluating tfidf implement.
# We implement tfidf from sklearn library.

from sklearn.feature_extraction.text import TfidfVectorizer

class tfidf(object):

    def __init__(self, emotion='joy'):
        self.emotion = emotion
        self.unique_word = None

    def reset_target(self, y):
        reset_y = [1 if i == self.emotion else 0 for i in y]
        return reset_y

    def extract_word(self, X):
        vectorizer = TfidfVectorizer()
        vectorizer.fit_transform(X)
        # fix the number of features from training data
        unique_word = vectorizer.get_feature_names_out()
        self.unique_word = unique_word

    def tfidf_matrix(self, X):
        # align the number of features between training data and testing data
        vectorizer = TfidfVectorizer(vocabulary=self.unique_word)
        tfidf_X = vectorizer.fit_transform(X)

        # turn the data return from .fit_transform() into matrix
        return tfidf_X.toarray()
