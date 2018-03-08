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
hlayers = [43, 73, (54, 38), (35, 26), (30, 11), (30, 17, 9), (63, 10)]

param_grid = [
    {'hidden_layer_sizes': hlayers, 'alpha': [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]}]

# metrics.OptimizeClassifier(data, target, MLPClassifier(solver='lbfgs', random_state=1), param_grid)
# Current Best: 0.931 (+/-0.068) for {'activation': 'relu', 'alpha': 0.001, 'hidden_layer_sizes': 73}
# 0.938 (+/-0.069) for {'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': 73}
# 0.951 (+/-0.057) for {'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': 43}
# 0.945 (+/-0.085) for {'activation': 'relu', 'alpha': 1e-06, 'hidden_layer_sizes': 43}
# 0.948 (+/-0.057) for {'alpha': 0.01, 'hidden_layer_sizes': (35, 26)}
# 0.942 (+/-0.074) for {'alpha': 5e-05, 'hidden_layer_sizes': (30, 11)}
# 0.952 (+/-0.057) for {'alpha': 0.0001, 'hidden_layer_sizes': (30, 11)}
# 0.947 (+/-0.036) for {'alpha': 0.001, 'hidden_layer_sizes': (30, 11)}
#

# temporary values to be replaced
ideal = [0, 0, 0]
maxi = 0.
# graph = np.zeros(hlayers.count)

# find the average F1 score and its standard deviation for all the layer sizes
print("hlayers/tp/tn/fp/fn/f1/precision/sensitivity/specificity")
for x in hlayers:
    temp = metrics.repeatedCrossValidatedScores(data, target,
                               MLPClassifier(solver='lbfgs', alpha=0.001, hidden_layer_sizes=x, random_state=1),
                               iterations=1)
    metrics.printAverages(x, temp)

    if np.average(temp['test_f1']) > maxi:
        maxi = np.average(temp['test_f1'])
        ideal = x

# print the best average and its F1 score
print(str(ideal) + " gives " + str(maxi *100) + "% accuracy")
#print("The standard deviation was " + str(maxi[1] * 100) + "%")
