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
               'Random Forest': RandomForestClassifier(n_estimators=250),
               'Logistic Regression': LogisticRegression(C=1000),
               'Neural Network': MLPClassifier(solver='lbfgs', alpha=0.001, hidden_layer_sizes=6, random_state=1),
               'Nearest Neighbours': KNeighborsClassifier(n_neighbors=1),
               'Decision Tree': tree.DecisionTreeClassifier()}
datasets = {'Support Vector Machine': metrics.preprocess(k=13, fsiter=1000),
            'Decision Tree': metrics.preprocess(k=15, fsiter=1000),
            'Random Forest': metrics.preprocess(k=20, fsiter=1000, scaling=False),
            'Logistic Regression': metrics.preprocess(k=10, fsiter=1000),
            'Neural Network': metrics.preprocess(k=8, fsiter=1000, scaling=False),
            'Nearest Neighbours': metrics.preprocess(k=10, fsiter=1000)}

''' k = 24
Support Vector Machine/72.1/75.32000000000001/2.54/0.04/98.07890501655206%/99.95277777777778%/96.55198412698412%/97.0876551226551%/98.26738095238093%
Decision Tree/71.92000000000002/71.58/3.56/2.94/95.59839408269745%/96.51919191919191%/95.21309523809522%/95.75873015873015%/95.67071428571425%
Random Forest/74.9/73.42/0.74/0.9400000000000001/98.89520320408863%/98.9161111111111%/99.02619047619048%/99.12507936507936%/98.88666666666666%
Logistic Regression/72.22/74.64/3.1/0.04/97.6825261013496%/99.95555555555555%/95.85317460317458%/96.4582178932179%/97.90511904761905%
Neural Network/72.92/74.1/2.0/0.9800000000000001/97.92791794892412%/98.84539682539682%/97.3079365079365%/97.69595238095239%/98.02166666666665%
Nearest Neighbours/69.82000000000001/75.58/4.6000000000000005/0.0/96.53088950265418%/100.0%/93.78650793650793%/94.89967532467534%/96.94380952380952%
'''

''' k = 16
Support Vector Machine/72.06/75.46/2.4800000000000004/0.0/98.19154002859882%/100.0%/96.67857142857143%/97.1280808080808%/98.34821428571428%
Decision Tree/71.48/72.64/3.2/2.6800000000000006/95.93810918699461%/96.80166666666666%/95.66349206349204%/96.29551226551227%/96.07464285714286%
Random Forest/73.3/74.86/0.5800000000000001/1.2600000000000002/98.76956492080329%/98.52212842712844%/99.19087301587301%/99.33611111111111%/98.76857142857143%
Logistic Regression/71.97999999999999/75.04/2.84/0.14/97.80898480604361%/99.84333333333333%/96.17857142857143%/96.76852813852811%/98.01488095238096%
Neural Network/70.72/74.82/2.8000000000000003/1.66/96.85703436742132%/97.97020202020204%/96.18174603174602%/96.77305916305914%/97.02535714285713%
Nearest Neighbours/72.72/74.8/2.4799999999999995/0.0/98.1744862980157%/100.0%/96.65873015873015%/97.12554112554113%/98.33678571428571%
'''
accuracies = []
sensitivities = []
specificities = []
for key in classifiers.keys():
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
plt.title('Relative Accuracy, Sensitivity, and Specificity of Algorithms')
plt.xlabel('Algorithm', size=8)
sens = ax.bar([1, 5, 9, 13, 17, 21], sensitivities, width=0.8, color='red', bottom=0.9)
acc = ax.bar([2, 6, 10, 14, 18, 22], accuracies, width=0.8, color='green', bottom=0.9)
spec = ax.bar([3, 7, 11, 15, 19, 23], specificities, width=0.8, color='blue', bottom=0.9)
plt.legend((sens, acc, spec), ('sensitivity', 'accuracy', 'specificity'), fontsize=9)
plt.show()
