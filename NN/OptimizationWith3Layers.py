import arff
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

# open the arff file
dataset = arff.load(open('ckd.arff'))

# pulls the data into a numpy array
raw_data = np.array(dataset['data'])

# takes everything except the last column
data = raw_data[:, :-1]

# just the last column
target = raw_data[:, -1]

# fixes missing data by taking values from other rows and taking the average
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

# this function takes the average of every column excluding the unknown values
imp.fit(data)

# inserts the average into the missing spots
data = imp.fit_transform(data)


# Function that creates the neural network 100 times and takes the average of its F1 score
def aveaccuracy(_data, _target, h1, h2, h3):
    toreturn = 0.
    for x in range(100):
        data_train, data_test, target_train, target_test = train_test_split(_data, _target, test_size=0.3)
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(h1, h2, h3), random_state=1)
        clf.fit(data_train, target_train)
        prediction = clf.predict(data_test)
        accuracy = f1_score(target_test, prediction, pos_label='1')
        toreturn += accuracy
    return toreturn


# default values to be overwritten
ideal = [0, 0, 0]
maxi = 0

# check a lot of hidden layer configurations for sets with high accuracy
for x in range(3, 20):
    for y in range(2, x):
        for z in range(1, y):
            temp = aveaccuracy(data, target, x, y, z)
            if temp > maxi:
                maxi = temp
                ideal = [x, y, z]
            if temp > 85:
                print("The predictions were " + str(temp) + "% accurate on average for " + str([x, y, z]))

# print the highest accuracy one
print(str(ideal) + " gives " + maxi + "% accuracy")
