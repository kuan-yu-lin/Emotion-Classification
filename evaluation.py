import numpy as np


def confusion_matrix(y_pred, y_true):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(0, len(y_pred)):
        if y_pred[i] == 1 and y_true[i] == 1:
            TP += 1
        elif y_pred[i] == 1 and y_true[i] == 0:
            FP += 1
        elif y_pred[i] == 0 and y_true[i] == 1:
            FN += 1
        elif y_pred[i] == 0 and y_true[i] == 0:
            TN += 1
    con_matrix = np.matrix([[TP, FP], [FN, TN]])
    return con_matrix


def precision(y_pred, y_true):
    matrix = confusion_matrix(y_pred, y_true)
    precision = matrix[0, 0] / (matrix[0, 0] + matrix[0, 1])
    return precision


def recall(y_pred, y_true):
    matrix = confusion_matrix(y_pred, y_true)
    recall = matrix[0, 0] / (matrix[0, 0] + matrix[1, 0])
    return recall


def f1score(y_pred, y_true):
    p = precision(y_pred, y_true)
    r = recall(y_pred, y_true)
    f1 = (2*p*r) / (p + r)
    return f1


'''
# This version doesn't work.
def f_score_eval(pred_values, actul_values):
    tp = len(list(set(pred_values) & set(actul_values)))
    fp = len(list(set(pred_values) - set(actul_values)))
    fn = len(list(set(actul_values) - set(pred_values)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f = 2 * precision * recall / (precision + recall)
    return precision, recall, f
'''