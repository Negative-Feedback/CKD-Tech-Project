import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import metrics

maxi = 0
ideal = (0, 0)

data, target = metrics.preprocess(k=8, fsiter=1000)
clf = KNeighborsClassifier(n_neighbors=1)
accuracy = 0
for x in range(1000):
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.1)
    print(x)

    clf.fit(data_train, target_train)
    predictions = clf.predict(data_test)
    tn, fp, fn, tp = (confusion_matrix(target_test, predictions).ravel())

    accuracy = accuracy + round((tp + tn)/(tp + tn + fp + fn), 2) / 10

print(accuracy)
