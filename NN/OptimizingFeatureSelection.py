import numpy as np
from sklearn.neural_network import MLPClassifier
import metrics
import warnings
warnings.filterwarnings("ignore")

# temporary values to be replaced
ideal = [0, 0, 0]
maxi = 0.
# graph = np.zeros(hlayers.count)

# find the average F1 score and its standard deviation for all the layer sizes
print("hlayers/tp/tn/fp/fn/f1/precision/sensitivity/specificity")
for x in range(1, 25):
    data, target = metrics.preprocess(k=x)
    temp = metrics.repeatedCrossValidatedScores(data, target,
                               MLPClassifier(solver='lbfgs', alpha=0.001, hidden_layer_sizes=43, random_state=1),
                               iterations=50, cv=10)
    metrics.printAverages(x, temp)

    if np.average(temp['test_f1']) > maxi:
        maxi = np.average(temp['test_f1'])
        ideal = x

# print the best average and its F1 score
print(str(ideal) + " gives " + str(maxi * 100) + "% accuracy")
#print("The standard deviation was " + str(maxi[1] * 100) + "%")
