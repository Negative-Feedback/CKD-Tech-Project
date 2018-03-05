import arff
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
import metrics
import warnings
warnings.filterwarnings("ignore")


# classification threshold

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

# 16, 14, 11 is the best so far, 6,3 was the best for 2 layers
hlayers = [43, 73, (10, 7, 5), 59, 76, (23, 21), (25, 16), (27, 5), (29, 27),
           (32, 19), (43, 10), (56, 41), (52, 15), (62, 40), (68, 13), (71, 50),
           (73, 14), (30, 17, 9)]

# temporary values to be replaced
ideal = [0, 0, 0]
maxi = 0
# graph = np.zeros(hlayers.count)

# find the average F1 score and its standard deviation for all the layer sizes
print("hlayers/tp/tn/fp/fn/f1/precision/sensitivity/specificity")
for x in hlayers:
    '''
    temp = metrics.crossValidatedScores(data, target,
                                        MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=x, random_state=1))
                                        '''
    temp = metrics.aveaccuracy(data, target,
                               MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=x, random_state=1),
                               iterations=20)
    metrics.printAverages(x, temp)

    if np.average(temp['test_f1']) > maxi:
        maxi = np.average(temp['test_f1'])
        ideal = x
    '''
    if temp['test_f1'] > maxi['test_f1']:
        maxi = temp
        ideal = x
    if temp['test_f1'] > 85:
        print("The predictions were " + str(temp['test_f1']) + "% accurate on average for " + str(x))
        #print("The standard deviation was " + str(temp[1] * 100) + "%")
    '''

# print the best average and its F1 score
print(str(ideal) + " gives " + str(maxi *100) + "% accuracy")
#print("The standard deviation was " + str(maxi[1] * 100) + "%")
