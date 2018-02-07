import arff as larff
import numpy as np
from scipy.io import arff
from sklearn import svm
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer

dataset = larff.load(open('C:/Users/Matthew/PycharmProjects/CKD-Tech-Project/chronic_kidney_disease.arff'))

raw_data = np.array(dataset['data'])


data = raw_data[:, :-1]
target = raw_data[:, -1]

#data = np.delete(data, -1, axis=1)

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(data)

data = imp.fit_transform(data)

#enc.fit(data)
#enc.fit(target)



#print(data.shape, target.shape)
#print(data)
#print(target)



data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.3)


clf = svm.SVC()
clf.fit(data_train, target_train)