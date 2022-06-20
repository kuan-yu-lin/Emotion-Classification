from perceptron_from_scratch import perceptron
from evaluation import confusion_matrix,  f1score
from tfidf import tfidf
import string
import pandas as pd

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

# load the training data
y_train, X_train = load_dataset('isear-train.txt')
# load the testing data
y_test, X_test = load_dataset('isear-test.txt')

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

'''
# The first version of the classifier.
# initialize perceptron models
p = perceptron()
p.fit(X_train_tm, y_train_b)
y_pred = p.predict(X_test_tm)

# evaluate the performance of the classifier
m = confusion_matrix(y_pred, y_test_b)
fscore = f1score(y_pred, y_test_b)
fscore_lst += [fscore]
print(m, fscore)
'''

# The second version of the classifier. We modify the parameters: learning rates and iterations.
# get the list of fscore of all different iterations and learning rates
fscore_lst = []

# The first part: get the result from different learning rates.
# make a list of all learning rates
learning_rate_lst = [0.01, 0.02, 0.03, 0.04, 0.05]
# iterate through the learning_rate_lst
for lr in learning_rate_lst:   
    # initialize the perceptron model
    p = perceptron(learning_rate=lr, n_iter=100)
    p.fit(X_train_tm, y_train_b)
    y_pred = p.predict(X_test_tm)

    # evaluate the performance of the classifier
    fscore = f1score(y_pred, y_test_b)
    fscore_lst += [fscore]

'''
# The second part: get the result from different numbers of iterations.
# iterate through the number of iterations of perceptron
for niter in range(1, 201):   
    # initialize the perceptron model
    p = perceptron(learning_rate=0.01, n_iter=niter)
    p.fit(X_train_tm, y_train_b)
    y_pred = p.predict(X_test_tm)

    # evaluate the performance of the classifier
    fscore = f1score(y_pred, y_test_b)
    fscore_lst += [fscore]
'''

# create the date frame for all f1 scores 
data = {'F1-score': fscore_lst}
df = pd.DataFrame(data)
df.index += 1
print(df)
