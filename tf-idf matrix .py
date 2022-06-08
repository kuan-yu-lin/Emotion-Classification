import string
from perceptron_from_scratch import perceptron
from evaluation import confusion_matrix,  f1score
from tfidf import tfidf
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

b = tfidf('guilt')
b.extract_word(X_train)
tfidfv = b.tfidf_matrix(X_train)
tfidfv_test = b.tfidf_matrix(X_test)
y_train = b.reset_target(y_train)
y_test = b.reset_target(y_test)

# evaluate the performance of the classifier
# m = confusion_matrix(y_pred, y_test)
# fscore = f1score(y_pred, y_test)
# fscore_lst += [fscore]
# print(m, fscore)

# get the fscore of all different iterations and learning rates
fscore_lst = []

# make a list of all learning rates
learning_rate_lst = [0.01, 0.02, 0.03, 0.04, 0.05]
# iterate through the learning_rate_lst
for lr in learning_rate_lst:   
    p = perceptron(learning_rate=lr, n_iter=100)
    p.fit(tfidfv, y_train)
    y_pred = p.predict(tfidfv_test)
    fscore = f1score(y_pred, y_test)
    fscore_lst += [fscore]

'''
# iterate through the number of iterations of perceptron
for niter in range(1, 201):   
    p = perceptron(learning_rate=0.01, n_iter=niter)
    p.fit(tfidfv, y_train)
    y_pred = p.predict(tfidfv_test)
    fscore = f1score(y_pred, y_test)
    fscore_lst += [fscore]
'''

# Create the date frame for all f1 scores by pandas
data = {'F1-score': fscore_lst}
df = pd.DataFrame(data)
df.index += 1
print(df)
