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

dataset = arff.load(open('C:/Users/gener/PycharmProjects/CKD-Tech-Project/chronic_kidney_disease.arff')) # loads the dataset
#change the filepath to where yours is
raw_data = np.array(dataset['data']) # pulls the data out into a numpy array

data = raw_data[:, :-1] # takes everything except the last column
target = raw_data[:, -1] # just the last column

imp = Imputer(missing_values='NaN', strategy='mean', axis=0) #fixes missing data by taking values from other rows and taking the average
imp.fit(data) #iirc this function takes the average
data = imp.fit_transform(data) #inserts the average into the missing spots
data, target = SMOTE().fit_sample(data, target) # oversamples the minority class (notckd)

#instantiating estimator object
knn = KNeighborsClassifier(n_neighbors = 5)
scores = cross_val_score(knn, data, target, cv=10, scoring='accuracy')

knn.fit(data,target)
print(scores)
print(scores.mean())
print(metrics.accuracy_score)

k_range = range(1,51)
param_grid = dict(n_neighbors=k_range)
print (param_grid)
grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy')
grid.fit(data, target)
grid_mean_scores_ = {result.mean_validation_score for result in grid.cv_results_}


plt.plot(k_range, grid_mean_scores_)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')


k_range = range (1,100)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    scores = metrics.repeatedCrossValidatedScores(data, target, knn, cv=10, iterations=10)
    k_scores.append(scores['test_accuracy'].mean())

plt.plot(k_range, k_scores)
plt.xlabel('Value of K')
plt.ylabel('Cross-Validation Accuracy')
plt.show()