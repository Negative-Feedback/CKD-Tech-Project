from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
import numpy as np

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

def crossValidatedScores(data, target, clf):
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.3)
    scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tp),
               'fp': make_scorer(fp), 'fn': make_scorer(fn),
               'f1': make_scorer(f1), 'precision': make_scorer(precision),
               'sensitivity': make_scorer(sensitivity), 'specificity': make_scorer(specificity)}
               #'ROC': make_scorer(roc)}
    results = cross_validate(clf.fit(data_train, target_train), data_test, target_test, scoring=scoring, cv=10)
    return results


def printAverages(x, a):
    print(str(x) + "/" + str(np.sum(a['test_tp'])) + "/" + str(np.sum(a['test_tn'])) + "/" + str(np.sum(a['test_fp']))
          + "/" + str(np.sum(a['test_fn'])) + "/" + str(np.average(a['test_f1']) * 100)
          + "%/" + str(np.average(a['test_precision']) * 100) + "%/" + str(np.average(a['test_sensitivity']) * 100)
          + "%/" + str(np.average(a['test_specificity']) * 100) + "%")