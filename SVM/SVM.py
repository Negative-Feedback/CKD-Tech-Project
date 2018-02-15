import arff
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.metrics import accuracy_score

dataset = arff.load(open('C:/Users/Matthew/PycharmProjects/CKD-Tech-Project/chronic_kidney_disease.arff')) # loads the dataset
#change the filepath to where yours is
raw_data = np.array(dataset['data']) # pulls the data out into a numpy array

data = raw_data[:, :-1] # takes everything except the last column
target = raw_data[:, -1] # just the last column

imp = Imputer(missing_values='NaN', strategy='mean', axis=0) #fixes missing data by taking values from other rows and taking the average
imp.fit(data) #iirc this fucntion takes the average
data = imp.fit_transform(data) #inserts the average into the missing spots

data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.3) #breaks the dataset into test and training data
#30% of data is test data

clf = svm.SVC()
clf.fit(data_train, target_train)
predicted = clf.predict(data_test)
print(accuracy_score(target_test, predicted))
