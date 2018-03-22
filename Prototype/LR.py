import arff
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import metrics
def main():
    data, target = metrics.preprocess(k=10, fsiter=1000)

    print("hlayers/tp/tn/fp/fn/f1/precision/sensitivity/specificity/accuracy")
    temp = metrics.repeatedCrossValidatedScores(data, target, LogisticRegression(C=1000), cv=10, iterations=50)  # Gives avaerage accuracy
    metrics.printAverages(1000,temp)

    model = LogisticRegression(C=1000) #Creates a copy of te function LogisticRegression and names it as model
    results = metrics.repeatedCrossValidatedScores(data, target, model, cv =10, iterations=50)#Gives avaerage accuracy
    print("Accuracy: %0.2f (+/- %0.2f)" % (results['test_accuracy'].mean()*100, results['test_accuracy'].std() * 200))#prints results

    return ("Accuracy: %0.2f (+/- %0.2f)" % (results['test_accuracy'].mean()*100, results['test_accuracy'].std() * 200))