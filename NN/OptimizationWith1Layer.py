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


# Function that creates the neural network 100 times and takes the average of its F1 score
def aveaccuracy(data, target, h1):
    toreturn = 0.
    for n in range(100):
        data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.3)
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=h1, random_state=1)
        clf.fit(data_train, target_train)
        prediction = clf.predict(data_test)
        accuracy = f1_score(target_test, prediction, pos_label='1')
        toreturn += accuracy
    return toreturn


# default values
ideal = [0]
maxi = 0

# check a lot of hidden layer configurations for sets with high accuracy
print("hlayers/tp/tn/fp/fn/f1/precision/sensitivity/specificity")
for x in range(1, 100):
        temp = metrics.crossValidatedScores(data, target,
                                        MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=x, random_state=1))
        metrics.printAverages(x, temp)
        '''
        if np.average(temp['test_f1']) > maxi:
            maxi = np.average(temp['test_f1'])
            ideal = [x]
        if np.average(temp['test_f1']) > 0.75:
            print("The predictions were " + str(np.average(temp['test_f1']) * 100) + "% accurate on average for " + str(x))
            '''

# print the highest accuracy one
print(str(ideal) + " gives " + str(maxi) + "% accuracy")
