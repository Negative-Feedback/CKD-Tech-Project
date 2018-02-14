import arff
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score

dataset = arff.load(open('ckd.arff'))
raw_data = np.array(dataset['data']) # pulls the data out into a numpy array

data = raw_data[:, :-1] # takes everything except the last column
target = raw_data[:, -1] # just the last column

imp = Imputer(missing_values='NaN', strategy='mean', axis=0) #fixes missing data by taking values from other rows and taking the average
imp.fit(data) #iirc this fucntion takes the average
data = imp.fit_transform(data) #inserts the average into the missing spots


def aveaccuracy(data, target, h):
    toreturn = 0.
    accuracy = np.zeros(100)
    for i in range(100):
        data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.3)
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=h, random_state=1)
        clf.fit(data_train, target_train)
        prediction = clf.predict(data_test)
        accuracy[i] = f1_score(target_test, prediction, pos_label='1')
        toreturn += accuracy[i]
    return [toreturn, np.std(accuracy)]


hlayers = [(6, 3), (13, 11), (14, 7), (23, 16), (24, 21), (10, 7, 6), (16, 14, 11), (17, 13, 2), (18, 4, 2),
           (18, 10, 7), (18, 12, 6), (18, 17, 11), (19, 6, 4), (20, 19, 5), (20, 19, 17), (21, 9, 5), (21, 13, 8),
           (21, 13, 12), (21, 17, 9), (22, 16, 8), (23, 22, 10), (24, 14, 13), (24, 16, 10), (24, 19, 9), (24, 19, 16),
           (24, 21, 5), (24, 22, 5)]

ideal = [0, 0, 0]
maxi = [0, 100]
for x in hlayers:
    temp = aveaccuracy(data, target, x)
    if temp[0] > maxi[0]:
        maxi = temp
        ideal = x
    if temp[0] > 85:
        print("The predictions were " + str(temp[0]) + "% accurate on average for " + str(x))
        print("The standard deviation was " + str(temp[1] * 100) + "%")

print(str(ideal) + " gives " + str(maxi[0]) + "% accuracy")
print("The standard deviation was " + str(maxi[1] * 100) + "%")
