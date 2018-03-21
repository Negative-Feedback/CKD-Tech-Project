import arff
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn import metrics
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import metrics
from sklearn import preprocessing

data , target = metrics.preprocess(k=8, fsiter=1000)


# param_grid = {"max_depth": [3, None],
#               "max_features": range(1, 11),
#               "min_samples_split": range(2, 11),
#               "min_samples_leaf": range(1, 11),
#               "bootstrap": [True, False]}
#
#
# metrics.OptimizeClassifier(data, target, RandomForestClassifier(), param_grid)


#instantiating estimator object
rf = RandomForestClassifier(n_estimators=250)
scores = metrics.repeatedCrossValidatedScores(data, target, rf, cv=10, iterations=100)
# rf.fit
print("title/tp/tn/fp/fn/f1/precision/sensitivity/specificity/accuracy")
metrics.printAverages("Random Forest", scores)
