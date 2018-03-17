import arff
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
import metrics
import warnings
warnings.filterwarnings("ignore")

data, target = metrics.preprocess(k=8, fsiter=1000, scaling=False)

# default values
ideal = [0]
maxi = 0

# check a lot of hidden layer configurations for sets with high accuracy
print("hlayers/tp/tn/fp/fn/f1/precision/sensitivity/specificity/accuracy")
for x in range(1, 100):
    temp = metrics.repeatedCrossValidatedScores(data, target,
                                                MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=x,
                                                              random_state=1,), iterations=20, cv=10)
    metrics.printAverages(x, temp)
    if np.average(temp['test_f1']) > maxi:
        maxi = np.average(temp['test_f1'])
        ideal = x

# print the highest accuracy one
print(str(ideal) + " gives " + str(maxi) + "% accuracy")
