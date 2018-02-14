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


def aveaccuracy(data, target, h1, h2, h3):
    toreturn = 0.
    for x in range(100):
        data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.3)
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(h1, h2, h3), random_state=1)
        clf.fit(data_train, target_train)
        prediction = clf.predict(data_test)
        accuracy = 0.
        for n in range(target_test.size):
            if target_test[n] == prediction[n]:
                accuracy += 1.
        accuracy /= target_test.size
        toreturn += accuracy
    return toreturn


ideal = [0, 0, 0]
maxi = 0
for x in range(3, 25):
    for y in range(2, x):
        for z in range(1, y):
            temp = aveaccuracy(data, target, x, y, z)
            if temp > maxi:
                maxi = temp
                ideal = [x, y, z]
            if temp > 85:
                print("The predictions were " + str(temp) + "% accurate on average for " + str([x, y, z]))

print(str(ideal) + " gives " + maxi + "% accuracy")