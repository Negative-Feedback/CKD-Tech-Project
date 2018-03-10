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

data, target = metrics.preprocess()

neuron_range = [516, 337, 710, 372, 814, 630, 790, 858, 502, 986, 616, 822, 351, 906, 467, 498, 735, 427, 830, 650,
                115, 411, 867, 904, 810, 321, 453, 618, 939, 660, 949, 673, 720, 719, 201, 956, 523, 760, 809, 992,
                1000, 547, 499, 875, 439, 982, 536, 581, 643, 521, 971, 566, 318, 924, 943, 605, 932, 787, 952, 755,
                829, 664, 682, 965, 870, 920, 128, 764, 418, 460, 996, 477, 578, 463, 696, 265, 711, 73, 541, 845,
                553, 747, 619, 890, 136, 967, 480, 727, 954, 211, 567, 150, 531, 743, 794, 349, 511, 577, 435, 162,
                637, 759, 409, 629, 43]
sens_range = []
neuron_accuracy = []
neuron_sensitivity = []
neuron_specificity = []
for x in neuron_range:
    temp = metrics.repeatedCrossValidatedScores(data, target,
                               MLPClassifier(solver='lbfgs', alpha=0.001, hidden_layer_sizes=x, random_state=1),
                               iterations=50, cv=10)
    metrics.printAverages(x, temp)
    # neuron_accuracy.append(np.average(temp['test_accuracy']))
    if (np.average(temp['test_sensitivity']) > 0.5) & (np.average(temp['test_specificity']) > 0.5):
        neuron_sensitivity.append(np.average(temp['test_sensitivity']))
        neuron_specificity.append(np.average(temp['test_specificity']))
        neuron_accuracy.append(np.average(temp['test_accuracy']))
        sens_range.append(x)

acc = plt.plot(sens_range, neuron_accuracy, label='accuracy')
sens = plt.plot(sens_range, neuron_sensitivity, label='sensitivity')
spec = plt.plot(sens_range, neuron_specificity, label='specificity')
plt.xlabel('Number of Neurons')
plt.ylabel('Cross-Validation Accuracy')
plt.grid = True
plt.show()
