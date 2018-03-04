import arff
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import Imputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics.scorer import make_scorer
from imblearn.over_sampling import SMOTE
import metrics


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
hlayers = [43, 73, (10, 7, 5), 59, 76]

# temporary values to be replaced
ideal = [0, 0, 0]
maxi = {'test_f1': 0}
# graph = np.zeros(hlayers.count)

# find the average F1 score and its standard deviation for all the layer sizes
print("hlayers/tp/tn/fp/fn/f1/precision/sensitivity/specificity")
for x in hlayers:
    temp = metrics.crossValidatedScores(data, target,
                                        MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=x, random_state=1))
    metrics.printAverages(x, temp)
    '''
    if temp['test_f1'] > maxi['test_f1']:
        maxi = temp
        ideal = x
    if temp['test_f1'] > 85:
        print("The predictions were " + str(temp['test_f1']) + "% accurate on average for " + str(x))
        #print("The standard deviation was " + str(temp[1] * 100) + "%")
    '''

# print the best average and its F1 score
print(str(ideal) + " gives " + str(maxi['test_f1']) + "% accuracy")
#print("The standard deviation was " + str(maxi[1] * 100) + "%")
