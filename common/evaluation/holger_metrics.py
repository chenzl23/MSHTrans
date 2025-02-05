def precision_score(y_pred, y_true):
    true_positive = sum(1 for true, pred in zip(y_true, y_pred) if true == pred == 1)
    true_negative = sum(1 for true, pred in zip(y_true, y_pred) if true == pred == 0)
    false_positive = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)
    false_negative = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)
    precision = ((true_positive) / 
                 (true_positive + false_positive)) if (true_positive + false_positive) else 0
    return precision

def recall_score(y_pred, y_true):
    true_positive = sum(1 for true, pred in zip(y_true, y_pred) if true == pred == 1)
    true_negative = sum(1 for true, pred in zip(y_true, y_pred) if true == pred == 0)
    false_positive = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)
    false_negative = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)
    recall = ((true_positive) / 
              (true_positive + false_negative)) if (true_positive + false_negative) else 0
    return recall

def accuracy_score(y_pred, y_true):
    true_positive = sum(1 for true, pred in zip(y_true, y_pred) if true == pred == 1)
    true_negative = sum(1 for true, pred in zip(y_true, y_pred) if true == pred == 0)
    false_positive = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)
    false_negative = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)
    accuracy = ((true_positive + true_negative) / 
              (true_positive + true_negative + false_positive + false_negative))
    return accuracy

def f1_score(y_pred, y_true):
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0