import arff
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import Imputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics.scorer import make_scorer
from imblearn.over_sampling import SMOTE

def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]
def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
def specificity(y_true, y_pred):
    return tn(y_true, y_pred) / (tn(y_true, y_pred) + fp(y_true, y_pred))

# open the arff file
dataset = arff.load(open('ckd.arff'))

# pulls the data into a numpy array
raw_data = np.array(dataset['data'])

# takes everything except the last column
data = raw_data[:, :-1]

# just the last column
target = raw_data[:, -1]
for x in target:
    x = int(x)

# fixes missing data by taking values from other rows and taking the average
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

# this function takes the average of every column excluding the unknown values
imp.fit(data)

# inserts the average into the missing spots
data = imp.fit_transform(data)

data, target = SMOTE().fit_sample(data, target)

# Function that creates the neural network 100 times and takes the average of its F1 score
def aveaccuracy(_data, _target, _hlayers):
    toreturn = 0.
    accuracy = np.zeros(100)
    for i in range(100):
        # split the data randomly
        data_train, data_test, target_train, target_test = train_test_split(_data, _target, test_size=0.3)

        # create the neural network and fit it to the data
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=_hlayers, random_state=1)
        clf.fit(data_train, target_train)

        # have the neural network predict the results and find its accuracy
        prediction = clf.predict(data_test)
        accuracy[i] = f1_score(target_test, prediction, pos_label='1')
        toreturn += accuracy[i]

        # return the accuracy and its standard deviation
    return [toreturn, np.std(accuracy)]

def crossValidatedScores(data, target, hlayers):
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.3)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=hlayers, random_state=1)
    scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tp),
               'fp': make_scorer(fp), 'fn': make_scorer(fn),
               'f1': 'f1', 'prec': 'precision', 'sensitivity': 'recall',
               'specificity': make_scorer(specificity)}
    results = cross_validate(clf.fit(data_train, target_train), data_test, target_test, scoring=scoring)
    return results

# 16, 14, 11 is the best so far, 6,3 was the best for 2 layers
hlayers = [43, 73, (10, 7, 5), 59, 76]

# temporary values to be replaced
ideal = [0, 0, 0]
maxi = [0, 100]
# graph = np.zeros(hlayers.count)

# find the average F1 score and its standard deviation for all the layer sizes
for x in hlayers:
    temp = crossValidatedScores(data, target, x)

    if temp['f1'] > maxi['f1']:
        maxi = temp
        ideal = x
    if temp['f1'] > 85:
        print("The predictions were " + str(temp['f1']) + "% accurate on average for " + str(x))
        print("The standard deviation was " + str(temp[1] * 100) + "%")

# print the best average and its F1 score
print(str(ideal) + " gives " + str(maxi[0]) + "% accuracy")
print("The standard deviation was " + str(maxi[1] * 100) + "%")
