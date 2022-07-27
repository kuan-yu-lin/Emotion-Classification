'''
Author: Kuan-Yu Lin
'''
import string
from perceptron import perceptron
from evaluation import confusion_matrix,  f1score
from BOW import BOW
from tfidf import tfidf
import os


# load dataset
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


if __name__ == "__main__":
    '''
    The ISEAR dataset we used can be found with the link:
    https://www.unige.ch/cisa/research/materials-and-online-research/research-material/
    After downloading the files, change the value of train_data_file_name to the name of the training data. Change the value of test_data_file_name to the name of the testing data.
    '''
    train_data_file_name = None
    test_data_file_name = None

    train_path = os.path.abspath(train_data_file_name)
    test_path = os.path.abspath(test_data_file_name)

    # load the data
    y_train, X_train = load_dataset(train_path)
    y_test, X_test = load_dataset(test_path)

    # select the emotion for classifier
    emotion = 'guilt'

    # initialize the BOG
    b = BOW(emotion)
    b.extract_word(X_train)
    # get term-frequency matrix
    tfv = b.term_freq_matrix(X_train)
    tfv_test = b.term_freq_matrix(X_test)
    # modify the labels in the datasets into binary for the target emotion
    y_train = b.reset_target(y_train)
    y_test = b.reset_target(y_test)

    # initialize tfidf
    b = tfidf(emotion)
    b.extract_word(X_train)
    # get tfidf matrix
    X_train_tm = b.tfidf_matrix(X_train)
    X_test_tm = b.tfidf_matrix(X_test)
    # modify the labels in the datasets into binary for the target emotion
    y_train_b = b.reset_target(y_train)
    y_test_b = b.reset_target(y_test)

    # initialize perceptron models
    p = perceptron()

    # apply the training data preprocessed by BOW
    p.fit(tfv, y_train)
    y_pred = p.predict(tfv_test)

    # apply the training data preprocessed by tfidf
    p.fit(X_train_tm, y_train_b)
    y_pred = p.predict(X_test_tm)

    # evaluate the performance of the classifier
    m_bow = confusion_matrix(y_pred, y_test_b)
    f1_bow = f1score(y_pred, y_test_b)

    # evaluate the performance of the classifier
    m_tfidf = confusion_matrix(y_pred, y_test)
    f1_tfidf = f1score(y_pred, y_test)

    print(f'This is the classifier for {emotion}')
    print('The confusion matrix of classifier with BOW: ', m_bow)
    print('The f1-score of classifier with BOW: ', f1_bow)
    print('The confusion matrix of classifier with tf-idf: ', m_tfidf)
    print('The f1-score of classifier with tf-idf: ', f1_tfidf)
