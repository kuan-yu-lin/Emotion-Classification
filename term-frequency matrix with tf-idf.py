import string
from perceptron_from_scratch import perceptron
from evaluation import confusion_matrix,  f1score
from BOG_tfidf import BOG_tfidf
import time

start_time = time.time()

# load the training data
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


y_train, X_train = load_dataset('isear-train.txt')
y_test, X_test = load_dataset('isear-test.txt')


b = BOG_tfidf('fear')
b.extract_word(X_train)
tfv = b.tf_idf_matrix(X_train)
tfv_test = b.tf_idf_matrix(X_test)
y_train = b.reset_target(y_train)
y_test = b.reset_target(y_test)


p = perceptron()
p.fit(tfv, y_train)
y_pred = p.predict(tfv_test)

# evaluate the performance of the classifier
m = confusion_matrix(y_pred, y_test)
fscore = f1score(y_pred, y_test)
print(m, fscore)

print("--- %s seconds ---" % (time.time() - start_time))