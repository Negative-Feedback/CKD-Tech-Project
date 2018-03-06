import arff
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
import metrics
import warnings
warnings.filterwarnings("ignore")
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

data, target = SMOTE().fit_sample(data, target)

# default values to be overwritten
ideal = [0, 0, 0]
maxi = 0
best = []

# check a lot of hidden layer configurations for sets with high accuracy
for x in range(3, 50):
    for y in range(2, x):
        for z in range(1, y):
            temp = metrics.aveaccuracy(data, target,
                                       MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(x, y, z), random_state=1),
                                       iterations=10)
            metrics.printAverages((x, y, z), temp)
            if np.average(temp['test_f1']) > maxi:
                maxi = np.average(temp['test_f1'])
                ideal = (x, y, z)
            if np.average(temp['test_f1']) > 0.8:
                best = np.append(best, [(x, y, z)])

# print the highest accuracy one
print(str(ideal) + " gives " + str(maxi) + "% accuracy")
for i in best:
    print(i)
