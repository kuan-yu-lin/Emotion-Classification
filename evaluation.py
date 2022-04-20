def f_score_eval(pred_values, actul_values):
    tp = len(list(set(pred_values) & set(actul_values)))
    fp = len(list(set(pred_values) - set(actul_values)))
    fn = len(list(set(actul_values) - set(pred_values)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f = 2 * precision * recall / (precision + recall)
    return f