from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import metrics
import numpy as np

data, target = metrics.preprocess(k=8, fsiter=1000)
clf = KNeighborsClassifier(n_neighbors=1)

scores = metrics.repeatedCrossValidatedScores(data, target, clf, iterations=16, cv=250)

metrics.printAverages('clf', scores)

clf.fit(data, target)

print(clf.predict([[0.75,       0.,         1.,         1.,         0.80952381, 0.75555556,
 0.,         0.        ]]))

joblib.dump(clf, 'classifier.pkl', compress=9)
