import arff
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.neural_network import MLPClassifier

dataset = arff.load(open('ckd.arff'))
raw_data = np.array(dataset['data']) # pulls the data out into a numpy array

data = raw_data[:, :-1] # takes everything except the last column
target = raw_data[:, -1] # just the last column

imp = Imputer(missing_values='NaN', strategy='mean', axis=0) #fixes missing data by taking values from other rows and taking the average
imp.fit(data) #iirc this fucntion takes the average
data = imp.fit_transform(data) #inserts the average into the missing spots

data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.3) #breaks the dataset into test and training data
#30% of data is test data

print(target_train.size)

aveAccuracy = 0.
for x in range(1000):
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.3)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(16, 14, 11), random_state=1)

    clf.fit(data_train, target_train)
    prediction = clf.predict(data_test)

    accuracy = 0.
    for n in range(target_test.size):
        if target_test[n] == prediction[n]:
            accuracy += 1.
    accuracy /= target_test.size
    print("The predictions were " + str(accuracy * 100.) + "% accurate")
    aveAccuracy += accuracy
aveAccuracy /= 10

print("The predictions were " + str(aveAccuracy) + "% accurate on average")
