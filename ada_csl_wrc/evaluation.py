from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn import metrics

def g_mean(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return np.sqrt((tp/(tp+fn))*(tn/(tn+fp)))

def evaluate(y_true:np.ndarray, y_pred:np.ndarray, cost_matrix:np.ndarray):
    """
    cost_matrix is a 2x2 matrix with the following structure:
    [[TN, FP],
     [FN, TP]]
     
    Should return: 
    Accuracy, Precision, Recall, F1, Cost
    """

    output = {}
    output["cost"] = (confusion_matrix(y_true, y_pred) * cost_matrix).sum()
    output["accuracy"] = accuracy_score(y_true, y_pred)
    output["precision"] = precision_score(y_true, y_pred)
    output["recall"] = recall_score(y_true, y_pred)
    output["f1"] = f1_score(y_true, y_pred)
    output["g_mean"] = g_mean(y_true, y_pred)
    return output

