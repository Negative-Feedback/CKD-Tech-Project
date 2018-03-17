import arff
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
import metrics
import warnings
warnings.filterwarnings("ignore")

data, target = metrics.preprocess(k=8, fsiter=1000, scaling=False)

# default values
ideal = [0, 0]
maxi = 0

# check a lot of hidden layer configurations for sets with high accuracy
for x in range(2, 75):
    for y in range(1, x):
        temp = metrics.repeatedCrossValidatedScores(data, target,
                                        MLPClassifier(solver='lbfgs', alpha=0.0001, hidden_layer_sizes=(x, y), random_state=1),
                                        cv=10,  iterations=10)
        metrics.printAverages((x, y), temp)
        if np.average(temp['test_f1']) > maxi:
            maxi = np.average(temp['test_f1'])
            ideal = [x, y]
# print the highest accuracy one
print(str(ideal) + " gives " + str(maxi) + "% accuracy")
