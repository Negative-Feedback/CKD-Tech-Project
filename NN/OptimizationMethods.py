from sklearn.neural_network import MLPClassifier
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
import metrics
import numpy as np


def crossValidatedScores(data, target, hlayers, clf):
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.3)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=hlayers, random_state=1)
    scoring = {'tp': make_scorer(metrics.tp), 'tn': make_scorer(metrics.tp),
               'fp': make_scorer(metrics.fp), 'fn': make_scorer(metrics.fn),
               'f1': make_scorer(metrics.f1), 'precision': make_scorer(metrics.precision),
               'sensitivity': make_scorer(metrics.sensitivity), 'specificity': make_scorer(metrics.specificity)}
               #'ROC': make_scorer(roc)}
    results = cross_validate(clf.fit(data_train, target_train), data_test, target_test, scoring=scoring, cv=10)
    return results


def printAverages(x, a):
    print(str(x) + "/" + str(np.average(a['test_tp'])) + " " + str(np.average(a['test_tn']))
          + " " + str(np.average(a['test_fp'])) + " " + str(np.average(a['test_fn']))
          + " " + str(np.average(a['test_f1'])) + " " + str(np.average(a['test_precision']))
          + " " + str(np.average(a['test_sensitivity'])) + " " + str(np.average(a['test_specificity'])))