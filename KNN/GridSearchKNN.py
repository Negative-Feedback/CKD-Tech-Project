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
minmax_scaler = preprocessing.MinMaxScaler (feature_range=(0,1))
data_minmax = minmax_scaler.fit_transform(data)

knn = KNeighborsClassifier()
k_range = range(1,8)
weight_options = ['uniform', 'distance']
algorithm_options = ['auto', 'ball_tree', 'kd_tree', 'brute']
p_options = range(1,3)

param_grid = {'n_neighbors': range(1,8), 'weights': ['uniform', 'distance'], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'], 'p': range(1,3)}
metrics.OptimizeClassifier(data_minmax, target, KNeighborsClassifier(), param_grid)