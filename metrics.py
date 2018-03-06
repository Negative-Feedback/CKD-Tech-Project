from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
import numpy as np

def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]
def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
def specificity(y_true, y_pred):
    if tn(y_true, y_pred) + fp(y_true, y_pred) == 0:
        return 0
    else:
        return tn(y_true, y_pred) / (tn(y_true, y_pred) + fp(y_true, y_pred))
def f1(y_true, y_pred):
    return f1_score(y_true, y_pred, pos_label='1')
def sensitivity(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label='1')
def precision(y_true, y_pred):
    return precision_score(y_true, y_pred, pos_label='1')
def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)
'''
def roc(y_true, y_pred):
    return roc_curve(y_true, y_pred, pos_label='1')
'''

def crossValidatedScores(data, target, clf, cv=3):
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.3)
    scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tn), 'fp': make_scorer(fp), 'fn': make_scorer(fn),
               'accuracy': make_scorer(accuracy), 'f1': make_scorer(f1), 'precision': make_scorer(precision),
               'sensitivity': make_scorer(sensitivity), 'specificity': make_scorer(specificity)}
               #'ROC': make_scorer(roc)}
    results = cross_validate(clf.fit(data_train, target_train), data_test, target_test, scoring=scoring, cv=cv)
    return results


def printAverages(x, a):
    print(str(x) + "/" + str(np.sum(a['test_tp'])) + "/" + str(np.sum(a['test_tn'])) + "/" + str(np.sum(a['test_fp']))
          + "/" + str(np.sum(a['test_fn'])) + "/" + str(np.average(a['test_f1']) * 100)
          + "%/" + str(np.average(a['test_precision']) * 100) + "%/" + str(np.average(a['test_sensitivity']) * 100)
          + "%/" + str(np.average(a['test_specificity']) * 100)
          + "%/" + str(np.average(a['test_accuracy']) * 100) + "%")


# Function that creates the neural network 100 times and takes the average of its F1 score
def repeatedCrossValidatedScores(_data, _target, _clf, iterations=50, cv=2):
    toreturn = crossValidatedScores(_data, _target, _clf, cv=cv)

    for i in range(iterations - 1):
        temp = crossValidatedScores(_data, _target, _clf, cv=cv)
        toreturn = {k: temp.get(k, 0) + toreturn.get(k, 0) for k in set(temp)}

    toReturn = {k: v / iterations for k, v in toreturn.items()}
    return toReturn
