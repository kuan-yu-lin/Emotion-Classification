import string
from perceptron_from_scratch import perceptron
from evaluation import confusion_matrix,  f1score
from BOW import BOW
import pandas as pd


# load function
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


# load training data
y_train, X_train = load_dataset('isear-train.txt')
# load test data
y_test, X_test = load_dataset('isear-test.txt')


# initialize the BOG for emotion == 'guilt'
b = BOW('sadness')
b.extract_word(X_train)
# get term-frequency matrix
tfv = b.term_freq_matrix(X_train)
# get tfm for training data
tfv_test = b.term_freq_matrix(X_test)
# modify the 'y's in the datasets so that it is binary for the target emotion
y_train = b.reset_target(y_train)
y_test = b.reset_target(y_test)

# The first version of the classifier.
# initialize perceptron models
p = perceptron()
p.fit(tfv, y_train)
y_pred = p.predict(tfv_test)

# The first version of the classifier.
# evaluate the performance of the classifier
m = confusion_matrix(y_pred, y_test)
fscore = f1score(y_pred, y_test)
print(m, fscore)

'''
# The second version of the classifier. We modify the parameters: learning rates and iterations.
# get the list of fscore of all results
fscore_lst = []

# The first part: get the result from different learning rates.
# make a list of all learning rates
learning_rate_lst = [0.01, 0.02, 0.03, 0.04, 0.05]
# iterate through the learning_rate_lst
for lr in learning_rate_lst:   
    # initialize the perceptron model
    p = perceptron(learning_rate=lr, n_iter=100)
    p.fit(tfv, y_train)
    y_pred = p.predict(tfv_test)

    # evaluate the performance of the classifier
    fscore = f1score(y_pred, y_test)
    fscore_lst += [fscore]


for niter in range(51, 71):
    # initialize the perceptron model   
    p = perceptron(learning_rate=0.01, n_iter=niter)
    p.fit(tfv, y_train)
    y_pred = p.predict(tfv_test)

    # evaluate the performance of the classifier
    fscore = f1score(y_pred, y_test)
    fscore_lst += [fscore]

# create the date frame for all f1 scores 
data = {'F1-score': fscore_lst}
df = pd.DataFrame(data)
df.index += 1
print(df)
'''