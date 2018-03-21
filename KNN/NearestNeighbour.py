import arff
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn import metrics
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

data, target = metrics.preprocess(k=8, fsiter=1000)

kn = KNeighborsClassifier(n_neighbors=1)
scores = metrics.repeatedCrossValidatedScores(data, target, kn, cv=10, iterations=500)

print("title/tp/tn/fp/fn/f1/precision/sensitivity/specificity/accuracy")
metrics.printAverages("K Nearest Neighbors", scores)