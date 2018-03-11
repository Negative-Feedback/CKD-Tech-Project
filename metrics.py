from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.over_sampling import SMOTE
import numpy as np
import warnings
import arff
warnings.simplefilter(action='ignore', category=FutureWarning)

def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]
def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
def specificity(y_true, y_pred):
    if tn(y_true, y_pred) + fp(y_true, y_pred) == 0:
        return 0
    else:
        return float(tn(y_true, y_pred)) / float((tn(y_true, y_pred) + fp(y_true, y_pred)))
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


def unisonshuffle(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    return a, b

def crossValidatedScores(data, target, clf, cv=3):
    scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tn), 'fp': make_scorer(fp), 'fn': make_scorer(fn),
               'accuracy': make_scorer(accuracy), 'f1': make_scorer(f1), 'precision': make_scorer(precision),
               'sensitivity': make_scorer(sensitivity), 'specificity': make_scorer(specificity)}
               #'ROC': make_scorer(roc)}

    results = cross_validate(clf, data, target, scoring=scoring, cv=cv, return_train_score=False)
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

    for i in range(0, iterations - 1):
        _data, _target = unisonshuffle(_data, _target)
        temp = crossValidatedScores(_data, _target, _clf, cv=cv)
        toreturn = {k: temp.get(k, 0) + toreturn.get(k, 0) for k in set(temp)}

    toreturn = {k: v / iterations for k, v in toreturn.items()}
    return toreturn


def OptimizeClassifier(data, target, clf, grid, scores={'f1': make_scorer(f1)}, cv=10, refit='f1'):
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.3)
    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(clf, grid, cv=cv,
                           scoring=scores, refit=refit)
        clf.fit(data_train, target_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_f1']
        stds = clf.cv_results_['std_test_f1']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.5f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = target_test, clf.predict(data_test)
        print(classification_report(y_true, y_pred))
        print()


def FeatureSelection(data, target, iterations=1000, columns=24):
    toReturn = np.zeros(columns, dtype=float)
    for x in range(iterations):
        test = ExtraTreesClassifier()
        test.fit(data, target)
        toReturn += test.feature_importances_
    for i in range(columns):
        toReturn[i] = toReturn[i] / float(iterations)
    return toReturn


def preprocess(k=24, fsiter=100, scaling=True):
    # open the arff file
    dataset = arff.load(open('chronic_kidney_disease.arff'))

    # pulls the data into a numpy array
    raw_data = np.array(dataset['data'])

    # takes everything except the last column
    data = raw_data[:, :-1]

    # just the last column
    target = raw_data[:, -1]

    # fixes missing data by taking values from other rows and taking the average
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

    # this function takes the average of every column excluding the unknown values
    imp.fit(data)

    # inserts the average into the missing spots
    data = imp.fit_transform(data)

    if scaling:
        minmax_scaler = MinMaxScaler(feature_range=(0, 1))
        data = minmax_scaler.fit_transform(data)

    # create a classifier to perform feature selection
    if k < 24:
        scores = FeatureSelection(data, target, iterations=fsiter)

        # remove unnecessary columns
        sortedScores = np.sort(scores)
        mask = np.ones(len(sortedScores), dtype=bool)
        for x in range(len(sortedScores)):
            if scores[x] < sortedScores[24-k]:
                mask[x] = False
        for x in range(23, -1, -1):
            if not mask[x]:
                data = np.delete(data, x, 1)
    data, target = SMOTE().fit_sample(data, target)
    return data, target
