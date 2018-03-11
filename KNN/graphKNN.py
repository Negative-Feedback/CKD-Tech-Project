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

dataset = arff.load(open('C:/Users/gener/PycharmProjects/CKD-Tech-Project/chronic_kidney_disease.arff')) # loads the dataset
#change the filepath to where yours is
raw_data = np.array(dataset['data']) # pulls the data out into a numpy array

data = raw_data[:, :-1] # takes everything except the last column
target = raw_data[:, -1] # just the last column

imp = Imputer(missing_values='NaN', strategy='mean', axis=0) #fixes missing data by taking values from other rows and taking the average
imp.fit(data) #iirc this function takes the average
data = imp.fit_transform(data) #inserts the average into the missing spots
data, target = SMOTE().fit_sample(data, target) # oversamples the minority class (notckd)
data_l1 = preprocessing.normalize(data, norm='l1')

k_range = range (1,26)
error_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    scores = metrics.repeatedCrossValidatedScores(data_l1, target, knn, cv=10, iterations=25)
    error_scores.append(1-scores['test_accuracy'].mean())
    print(k)

plt.plot(k_range, error_scores)
plt.xlabel('Number of Neighbours(K)')
plt.ylabel('% error')
plt.show()