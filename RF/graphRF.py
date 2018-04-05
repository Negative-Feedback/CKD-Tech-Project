from sklearn.ensemble import RandomForestClassifier
import metrics
import matplotlib.pyplot as plt

data , target = metrics.preprocess(k=13, fsiter=1000)

C_range = range (64, 258, 8)
accuracy_scores = []
for c in C_range:
    rf = RandomForestClassifier(n_estimators=c)
    scores = metrics.repeatedCrossValidatedScores(data, target, rf, cv=10, iterations=50)
    temp = scores['test_accuracy'].mean()
    if temp > 0.9:
        accuracy_scores.append(temp)
    metrics.printAverages(c, scores)

plt.plot(C_range, accuracy_scores)
plt.title('Random Forest Optimization', size=11, fontweight='bold')
plt.xlabel('Number of Estimators', size=8)
plt.ylabel('Accuracy', size=8)
plt.show()
