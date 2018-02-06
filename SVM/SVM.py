from scipy.io import arff
from sklearn import svm
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import datasets

clf = svm.SVC()


raw_data, meta = arff.loadarff('C:/Users/Matthew/PycharmProjects/CKD-Tech-Project/chronic_kidney_disease.arff')

print(raw_data.shape)
data = raw_data[meta.names()[:-1]] #everything but the last column
target = raw_data[meta.names()[-1:]]  #just the last column

#train_data = train_data.view(np.float).reshape(data.shape + (-1,)) #converts the record array to a normal numpy array

data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.3)

clf.fit(data_train, target_train)

