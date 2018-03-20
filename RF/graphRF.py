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

data , target = metrics.preprocess(k=10, fsiter=1000)

C_range = range (1,501)
accuracy_scores = []
for c in C_range:
    rf = RandomForestClassifier(n_estimators= c)
    scores = metrics.repeatedCrossValidatedScores(data, target, rf, cv=10, iterations=1)
    accuracy_scores.append(scores['test_accuracy'].mean())
    print(c)

plt.plot(C_range, accuracy_scores)
plt.title('RF vs. % Error', size=11, fontweight='bold')
plt.xlabel('Number of Neighbours(K)', size=8)
plt.ylabel('% Error', size=8)
plt.show()
