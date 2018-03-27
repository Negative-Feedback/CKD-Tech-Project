from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import metrics

data, target = metrics.preprocess(k=8, fsiter=1000)
clf = KNeighborsClassifier(n_neighbors=1)

scores = metrics.repeatedCrossValidatedScores(data, target, clf, iterations=1000, cv=10)

metrics.printAverages('clf', scores)

clf.fit(data, target)

joblib.dump(clf, 'classifier.pkl', compress=9)
