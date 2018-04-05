import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

data, target = metrics.preprocess()
classifiers = {'Support Vector Machine': svm.SVC(C=1, kernel='linear', decision_function_shape='ovo', random_state=6),
               'Random Forest': RandomForestClassifier(n_estimators=184),
               'Logistic Regression': LogisticRegression(C=1000),
               'Nearest Neighbours': KNeighborsClassifier(n_neighbors=1),
               'Decision Tree': tree.DecisionTreeClassifier(),
               'Neural Network': MLPClassifier(solver='lbfgs', alpha=0.001, hidden_layer_sizes=54, random_state=1)}
datasets = {'Support Vector Machine': metrics.preprocess(k=19, fsiter=1000),
            'Decision Tree': metrics.preprocess(k=6, fsiter=1000, scaling=False),
            'Random Forest': metrics.preprocess(k=13, fsiter=1000, scaling=False),
            'Logistic Regression': metrics.preprocess(k=11, fsiter=1000),
            'Neural Network': metrics.preprocess(k=8, fsiter=1000),
            'Nearest Neighbours': metrics.preprocess(k=8, fsiter=1000)}

accuracies = []
sensitivities = []
specificities = []
for key in classifiers.keys():
    print(key)
    data, target = datasets[key]
    temp = metrics.repeatedCrossValidatedScores(data, target, classifiers[key], iterations=100, cv=10)
    metrics.printAverages(key, temp)
    accuracies.append(np.average(temp['test_accuracy']) - 0.9)
    sensitivities.append(np.average(temp['test_sensitivity']) - 0.9)
    specificities.append(np.average(temp['test_specificity']) - 0.9)

plt.figure()
ax = plt.subplot()
plt.xticks([2, 6, 10, 14, 18, 22], classifiers.keys(), size=5.5)
plt.yticks([0.9, 0.925, 0.95, 0.975, 1.0], ["90%", "92.5%", "95%", "97.5%", "100%"])
plt.title('Relative Success with optimal features')
plt.xlabel('Algorithm', size=8)
sens = ax.bar([1, 5, 9, 13, 17, 21], sensitivities, width=0.8, color='red', bottom=0.9)
acc = ax.bar([2, 6, 10, 14, 18, 22], accuracies, width=0.8, color='green', bottom=0.9)
spec = ax.bar([3, 7, 11, 15, 19, 23], specificities, width=0.8, color='blue', bottom=0.9)
plt.show()
