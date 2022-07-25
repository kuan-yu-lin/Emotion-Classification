from perceptron_from_scratch import perceptron
from evaluation import confusion_matrix,  f1score
from tfidf import tfidf
import string
import os

# define a function for loading the data
def load_dataset(data):

    with open(data, mode='r', encoding='utf-8') as f:
        rows = [l.strip().split('\t')[:2] for l in f]
    y = []
    X = []
    for row in rows:
        if len(row) == 2:
            y.append(row[0])
            X.append(row[1].translate(str.maketrans(
                '', '', string.punctuation)).lower().split())
    return y, X

train_path = os.path.abspath('data/train.txt')
test_path = os.path.abspath('data/test.txt')

# load the training data
y_train, X_train = load_dataset(train_path)
# load the testing data
y_test, X_test = load_dataset(test_path)

# initialize tfidf with one emotion
b = tfidf('guilt')
# get the number of feature from training data
b.extract_word(X_train)

# get tfidf matrix for training data
X_train_tm = b.tfidf_matrix(X_train)
# get tfidf matrix for testing data
X_test_tm = b.tfidf_matrix(X_test)
# modify the 'y's in the datasets into binary for the target emotion
y_train_b = b.reset_target(y_train)
y_test_b = b.reset_target(y_test)

# initialize perceptron models
p = perceptron()
p.fit(X_train_tm, y_train_b)
y_pred = p.predict(X_test_tm)

# evaluate the performance of the classifier
m = confusion_matrix(y_pred, y_test_b)
fscore = f1score(y_pred, y_test_b)
print(m, fscore)
