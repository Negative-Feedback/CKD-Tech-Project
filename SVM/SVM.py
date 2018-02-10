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





c_val = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
tol_val = [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.011,0.012,0.013,0.014,0.015]
shape = ['ovo','ovr']
rstate = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

for i in range (0,14):
    total = 0
    for j in range (0,15):
        for k in range (0,2):
            for l in range (0,15):
                for m in range (0, 25):
                    clf = svm.SVC(C=c_val[i], kernel='linear', tol = tol_val[j], decision_function_shape=shape[k], random_state=rstate[l])
                    clf.fit(data_train, target_train)
                    predicted = clf.predict(data_test)
                    total += accuracy_score(target_test, predicted) * 100

    print('Accuracy: '+ '%.3f'%(total/25) + 'C: '+ c_val[i] + 'tolerance: '+tol_val[j] + 'shape: '+ shape[k] + 'state: '+rstate[l])








total = 0
for x in range(0, 100):
    clf = svm.SVC(C=1.5, kernel='sigmoid', decision_function_shape='ovo', random_state= 1)
    clf.fit(data_train, target_train)
    predicted = clf.predict(data_test)
    total += accuracy_score(target_test, predicted) *100

total /= 100


print(np.array_equal(predicted, target_test))
print(total)