import numpy as np
from sklearn.ensemble import RandomForestClassifier
import metrics
import matplotlib.pyplot as plt

maxi = 0
ideal = (0, 0)

num_features = []
accs = []

for features in range(1, 25):
    data, target = metrics.preprocess(k=features, fsiter=100)
    temp = metrics.repeatedCrossValidatedScores(data, target,
                                                RandomForestClassifier(n_estimators=128),
                                                iterations=100, cv=10)
    metrics.printAverages(features, temp)

    num_features.append(features)
    accs.append(np.average(temp['test_accuracy']))

print(str(ideal) + " gives " + str(maxi) + "% accuracy")

acc, = plt.plot(num_features, accs, label='Accuracy')
plt.title("Feature Selection for Random Forest", fontsize=14)
plt.xlabel('Number of Features')
plt.ylabel('Repeated-Cross-Validation Accuracy (%)')
plt.yticks([0.85, 0.90, 0.95, 1], ["85", "90", "95", "100"])
plt.xticks([0, 4, 8, 12, 16, 20, 24])
plt.show()
