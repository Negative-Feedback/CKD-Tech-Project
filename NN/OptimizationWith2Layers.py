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

# fixes missing data by taking values from other rows and taking the average
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

# this function takes the average of every column excluding the unknown values
imp.fit(data)

# inserts the average into the missing spots
data = imp.fit_transform(data)

data, target = SMOTE().fit_sample(data, target)


# Function that creates the neural network 100 times and takes the average of its F1 score
def aveaccuracy(data, target, h1, h2):
    toreturn = 0.
    for n in range(50):
        data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.3)
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(h1, h2), random_state=1)
        clf.fit(data_train, target_train)
        prediction = clf.predict(data_test)
        accuracy = f1_score(target_test, prediction, pos_label='1')
        toreturn += accuracy
    toreturn *= 2
    return toreturn

def crossValidatedScores(data, target, hlayers):
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.3)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=hlayers, random_state=1)
    scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tp),
               'fp': make_scorer(fp), 'fn': make_scorer(fn),
               'f1': 'f1', 'prec': 'precision', 'sensitivity': 'recall',
               'specificity': make_scorer(specificity)}
    results = cross_validate(clf.fit(data_train, target_train), data_test, target_test, scoring=scoring)
    return results



# default values
ideal = [0, 0]
maxi = 0

# check a lot of hidden layer configurations for sets with high accuracy
for x in range(2, 50):
    for y in range(1, x):
        temp = crossValidatedScores(data, target, (x, y))
        if temp['f1'] > maxi:
            maxi = temp
            ideal = [x, y]
        if temp > 84:
            print("The predictions were " + str(temp) + "% accurate on average for " + str([x, y]))

# print the highest accuracy one
print(str(ideal) + " gives " + str(maxi) + "% accuracy")
