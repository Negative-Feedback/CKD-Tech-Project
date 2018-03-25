import metrics
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier


data, target = metrics.preprocess(k=8, fsiter=1000)
target = label_binarize(target, classes=["0", "1"]).ravel()

data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=.9,)

# Learn to predict each class against the other
classifier = KNeighborsClassifier(n_neighbors=1)
target_score = classifier.fit(data_train, target_train).predict(data_test)

# Compute ROC curve and ROC area for each class

fpr, tpr, _ = roc_curve(target_test, target_score)
roc_auc = auc(fpr, tpr)

# Compute micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(target_test.ravel(), target_score.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
