import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import arff
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.neural_network import MLPClassifier
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

# alpha_range = [float(i) / 100000 for i in range(1, 1001)]
alpha_range = [0.1, 0.01, 0.001, 0.0001, 1e-5]
alpha_accuracy = []
alpha_sensitivity = []
alpha_specificity = []
for x in alpha_range:
    temp = metrics.repeatedCrossValidatedScores(data, target,
                               MLPClassifier(solver='lbfgs', alpha=x, hidden_layer_sizes=43, random_state=1),
                               iterations=50, cv=10)
    metrics.printAverages('%.5f' % x, temp)
    alpha_accuracy.append(np.average(temp['test_accuracy']))
    # alpha_sensitivity.append(np.average(temp['test_sensitivity']))
    # alpha_specificity.append(np.average(temp['test_specificity']))

plt.plot(range(1, 6), alpha_accuracy)
plt.xlabel('Value of Alpha')
plt.ylabel('Cross-Validation Accuracy')
plt.grid = True
plt.show()