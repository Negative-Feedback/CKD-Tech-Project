import arff
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn import metrics
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import metrics
import graphviz
from sklearn.tree import export_graphviz

data , target = metrics.preprocess(k=10, fsiter=1000)

#instantiating estimator object
dt = DecisionTreeClassifier()
scores = metrics.repeatedCrossValidatedScores(data, target, dt, cv=10, iterations=100)
# rf.fit
print("title/tp/tn/fp/fn/f1/precision/sensitivity/specificity/accuracy")
metrics.printAverages("Decision Tree", scores)