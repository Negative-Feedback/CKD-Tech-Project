import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import metrics

maxi = 0
ideal = (0, 0)

for features in range(1, 24):
    data, target = metrics.preprocess(k=features, fsiter=1000)
    for neighbours in [1, 3]:
        temp = metrics.repeatedCrossValidatedScores(data, target,
                                                    KNeighborsClassifier(n_neighbors=neighbours),
                                                    iterations=100, cv=10)
        metrics.printAverages((features, neighbours), temp)
        if np.average(temp['test_f1']) > maxi:
            maxi = np.average(temp['test_f1'])
            ideal = (features, neighbours)

print(str(ideal) + " gives " + str(maxi) + "% accuracy")