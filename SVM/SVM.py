#liac-arff, numpy, scipy, and scikit-learn are needed to run this
import arff
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

dataset = arff.load(open('C:/Users/Matthew/PycharmProjects/CKD-Tech-Project/chronic_kidney_disease.arff')) # loads the dataset
#change the filepath to where yours is

raw_data = np.array(dataset['data']) # pulls the data out into a numpy array

data = raw_data[:, :-1] # takes everything except the last column
target = raw_data[:, -1] # just the last column



imp = Imputer(missing_values='NaN', strategy='mean', axis=0) #fixes missing data by taking values from other rows and taking the average
imp.fit(data) #iirc this fucntion takes the average
data = imp.fit_transform(data) #inserts the average into the missing spots
data, target = SMOTE().fit_sample(data, target)

data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.3)
 #breaks the dataset into test and training data
#30% of data is test data


total = 0 # counter to hold the results of all the runs
run = 100 # this makes it so that i can adjust the how man times the loop runs with out manually changing what temp is divided by

#this for loop is used to average get an average accuracy
#For some reason we get slightly different accuracies each time we run the same thing. we are looking into why this happens
#so this loop allows us to know if our changes increase or decrease accuracy
for x in range(0, run):
    clf = svm.SVC(C = 1, kernel='linear', decision_function_shape='ovo', random_state= 6)
    clf.fit(data_train, target_train)
    predicted = clf.predict(data_test)
    total += accuracy_score(target_test, predicted) *100

total /= run


print(total)