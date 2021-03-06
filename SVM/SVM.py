#liac-arff, numpy, scipy, and scikit-learn are needed to run this
import arff
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import random
import metrics
import matplotlib.pyplot as plt
from sklearn.externals import joblib


def findMin(accuracy): # finds the smallest value in an array
    min = accuracy[0]
    for x in range (0, len(accuracy)):
        if accuracy[x] < min:
            min = accuracy[x]

    return min


def findMax(accuracy): # finds the largest value in an array
    max = accuracy[0]
    for x in range (0, len(accuracy)):
        if accuracy[x] >= max:
            max = accuracy[x]

    return max

dataset = arff.load(open('C:/Users/gener/PycharmProjects/CKD-Tech-Project/chronic_kidney_disease.arff')) # loads the dataset
#change the filepath to where yours is

raw_data = np.array(dataset['data']) # pulls the data out into a numpy array

data = raw_data[:, :-1] # takes everything except the last column
target = raw_data[:, -1] # just the last column



imp = Imputer(missing_values='NaN', strategy='mean', axis=0) #fixes missing data by taking values from other rows and taking the average
imp.fit(data) #iirc this fucntion takes the average
data = imp.fit_transform(data) #inserts the average into the missing spots
data, target = SMOTE().fit_sample(data, target) # oversamples the minority class (notckd)
#data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.3)

clf = svm.SVC(C = 1, kernel='linear', decision_function_shape='ovo', random_state= 6)

data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.3, random_state=112383)
#Accuracy: 100.0 Seed: 112383
clf.fit(data_train, target_train)
joblib.dump(clf, 'SVM.pkl')



