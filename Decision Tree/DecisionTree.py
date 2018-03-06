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
dt = DecisionTreeClassifier()
scores = cross_val_score(dt, data, target, cv=10, scoring='accuracy')
dt.fit(data,target)
print(scores)
print(scores.mean())
print(metrics.accuracy_score)