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

data, target = metrics.preprocess(k=8, fsiter=1000)

k_range = range(1, 26)
error_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = metrics.repeatedCrossValidatedScores(data, target, knn, cv=10, iterations=100)
    error_scores.append((1-scores['test_accuracy'].mean()) * 100)
    print(k)

plt.plot(k_range, error_scores)
plt.title("Number of Neighbours for KNN", fontsize=14)
plt.xlabel('Number of Neighbours(K)')
plt.ylabel('% Error')
plt.show()