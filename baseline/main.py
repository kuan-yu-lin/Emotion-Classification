'''
Authors: Tzu-Ju Lin and Kuan-Yu Lin
'''
import string
from perceptron import perceptron
from evaluation import confusion_matrix, f1score, precision, recall
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

    train_data_file_name = './data/train.txt'
    test_data_file_name = './data/test.txt'

    train_path = os.path.abspath(train_data_file_name)
    test_path = os.path.abspath(test_data_file_name)

    # load the data
    y_train, X_train = load_dataset(train_path)
    y_test, X_test = load_dataset(test_path)

    # select the emotion for classifier
    emotion = input('Which emotion you want to classify?')

    # initialize perceptron models
    p = perceptron()

    '''
    Perceptron with BOW
    '''
    # initialize the BOW
    b = BOW(emotion)
    b.extract_word(X_train)
    # get term-frequency matrix
    X_train_b = b.term_freq_matrix(X_train)
    X_test_b = b.term_freq_matrix(X_test)
    # modify the labels in the datasets into binary for the target emotion
    y_train_b = b.reset_target(y_train)
    y_test_b = b.reset_target(y_test)

    # apply the training data preprocessed by BOW
    p.fit(X_train_b, y_train_b)
    y_pred_b = p.predict(X_test_b)

    # evaluate the performance of the classifier
    cm_bow = confusion_matrix(y_pred_b, y_test_b)
    precision_bow = precision(y_pred_b, y_test_b)
    recall_bow = recall(y_pred_b, y_test_b)
    f1_bow = f1score(y_pred_b, y_test_b)

    '''
    Perceptron with tfidf
    '''
    # initialize tfidf
    t = tfidf(emotion)
    t.extract_word(X_train)
    # get tfidf matrix
    X_train_t = t.tfidf_matrix(X_train)
    X_test_t = t.tfidf_matrix(X_test)
    # modify the labels in the datasets into binary for the target emotion
    y_train_t = t.reset_target(y_train)
    y_test_t = t.reset_target(y_test)
  
    # apply the training data preprocessed by tfidf
    p.fit(X_train_t, y_train_t)
    y_pred_t = p.predict(X_test_t)

    # evaluate the performance of the classifier
    cm_tfidf = confusion_matrix(y_pred_t, y_test_t)
    precision_tfidf = precision(y_pred_t, y_test_t)
    recall_tfidf = recall(y_pred_t, y_test_t)
    f1_tfidf = f1score(y_pred_t, y_test_t)
    
    # produce the results
    print(f'This is the classifier for {emotion}.')
    print('the results of perceptron with BOW ----')
    print('confusion matrix: ')
    print(cm_bow)
    print('precision: ', precision_bow)
    print('recall: ', recall_bow)
    print('f1-score: ', f1_bow)
    print('the results of perceptron with tfidf ----')
    print('confusion matrix: ')
    print(cm_tfidf)
    print('precision: ', precision_tfidf)
    print('recall: ', recall_tfidf)
    print('f1-score: ', f1_tfidf)
