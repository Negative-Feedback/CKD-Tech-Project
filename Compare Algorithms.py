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
               'Decision Tree': tree.DecisionTreeClassifier(),
               'Random Forest': RandomForestClassifier(n_estimators=250),
               'Logistic Regression': LogisticRegression(C=1000),
               'Neural Network': MLPClassifier(solver='lbfgs', alpha=0.001, hidden_layer_sizes=73, random_state=1),
               'Nearest Neighbours': KNeighborsClassifier(n_neighbors=3)}

accuracies = []
sensitivities = []
specificities = []
for key in classifiers.keys():
    temp = metrics.repeatedCrossValidatedScores(data, target, classifiers[key], iterations=50, cv=10)
    metrics.printAverages(key, temp)
    accuracies.append(np.average(temp['test_accuracy']) - 0.5)
    sensitivities.append(np.average(temp['test_sensitivity']) - 0.5)
    specificities.append(np.average(temp['test_specificity']) - 0.5)

plt.figure()
ax = plt.subplot()
plt.xticks([2, 6, 10, 14, 18, 22], classifiers.keys(), size=5.5)
plt.yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], ["50%", "60%", "70%", "80%", "90%", "100%"])
plt.title('Relative Accuracy, Sensitivity, and Specificity of Algorithms')
plt.xlabel('Algorithm', size=8)
sens = ax.bar([1, 5, 9, 13, 17, 21], sensitivities, width=0.8, color='red', bottom=0.5)
acc = ax.bar([2, 6, 10, 14, 18, 22], accuracies, width=0.8, color='green', bottom=0.5)
spec = ax.bar([3, 7, 11, 15, 19, 23], specificities, width=0.8, color='blue', bottom=0.5)
plt.legend((sens, acc, spec), ('sensitivity', 'accuracy', 'specificity'), fontsize=7)
plt.show()
