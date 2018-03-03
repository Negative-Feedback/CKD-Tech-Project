from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred, labels=['0', '1'])[1, 1]
def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
def specificity(y_true, y_pred):
    return tn(y_true, y_pred) / (tn(y_true, y_pred) + fp(y_true, y_pred))
def f1(y_true, y_pred):
    return f1_score(y_true, y_pred, pos_label='1')
def sensitivity(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label='1')
def precision(y_true, y_pred):
    return precision_score(y_true, y_pred, pos_label='1')
'''
def roc(y_true, y_pred):
    return roc_curve(y_true, y_pred, pos_label='1')
'''