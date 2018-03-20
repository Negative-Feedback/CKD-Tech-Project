import matplotlib.pyplot as plt
import numpy as np
import metrics
from sklearn.neural_network import MLPClassifier

sens_range = []
neuron_accuracy = []
for features in range(1, 25):
    data, target = metrics.preprocess(k=features, fsiter=1000)
    maxacc = 0
    for neuron in range(40, 60):
        temp = metrics.repeatedCrossValidatedScores(data, target,
                               MLPClassifier(solver='lbfgs', alpha=0.001, hidden_layer_sizes=neuron, random_state=1),
                               iterations=50, cv=10)
        metrics.printAverages((features, neuron), temp)
        if np.average(temp['test_accuracy']) > maxacc:
            maxacc = np.average(temp['test_accuracy'])
    neuron_accuracy.append(np.average(temp['test_accuracy']))
    sens_range.append(features)

for x in range(np.size(sens_range)):
    print((sens_range[x], neuron_accuracy[x]))
acc, = plt.plot(sens_range, neuron_accuracy, label='Accuracy')
plt.title("Feature Selection for Neural Networks", fontsize=14)
plt.xlabel('Number of Features')
plt.ylabel('Maximum Repeated-Cross-Validation Accuracy (%)')
plt.yticks([0.85, 0.90, 0.95, 1], ["85", "90", "95", "100"])
plt.xticks([0, 4, 8, 12, 16, 20, 24])
plt.show()
