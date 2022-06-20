# This is the version for evaluating tfidf implement.
# We implement tfidf, perceptron and f1_score from sklearn library.


from sklearn.linear_model import Perceptron
from tfidf_newForEval import tfidf
from sklearn.metrics import f1_score
import os.path


# define a function for loading the data
def load_dataset(data):

    with open(data, mode='r', encoding='utf-8') as f:
        rows = [l.strip().split('\t')[:2] for l in f]
    y = []
    X = []
    for row in rows:
        if len(row) == 2:
            y.append(row[0])
            X.append(row[1])
    return y, X


# load the training data
y_train, X_train = load_dataset(os.path.dirname(__file__) + '/../isear-train.txt')
# load the testing data
y_test, X_test = load_dataset(os.path.dirname(__file__) + '/../isear-test.txt')

# initialize tfidf with one emotion
b = tfidf('joy')
# get the number of feature from training data
b.extract_word(X_train)

# get tfidf matrix for training data
X_train_tm = b.tfidf_matrix(X_train)
# get tfidf matrix for testing data
X_test_tm = b.tfidf_matrix(X_test)
# modify the 'y's in the datasets into binary for the target emotion
y_train_b = b.reset_target(y_train)
y_test_b = b.reset_target(y_test)

# initialize the perceptron model
p = Perceptron()
p.fit(X_train_tm, y_train_b)
y_pred = p.predict(X_test_tm)

# evaluate the performance of the classifier
fscore = f1_score(y_pred, y_test_b)
print(fscore)
