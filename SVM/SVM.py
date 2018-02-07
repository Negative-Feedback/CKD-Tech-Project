import arff as larff
import numpy as np
from scipy.io import arff
from sklearn import svm
from sklearn.model_selection import train_test_split

raw_data, meta = arff.loadarff('C:/Users/Matthew/PycharmProjects/CKD-Tech-Project/chronic_kidney_disease.arff')
target = raw_data[meta.names()[-1:]]  #just the last column

dataset = larff.load(open('C:/Users/Matthew/PycharmProjects/CKD-Tech-Project/chronic_kidney_disease.arff'))
data = np.array(dataset['data'])

data = np.delete(data, -1, axis=1)

#print(data.shape, target.shape)
#print(data)

target = np.asarray(target)

data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.3)

clf = svm.SVC()
clf.fit(data_train, target_train)