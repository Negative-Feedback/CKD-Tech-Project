import arff
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn import metrics
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import metrics

dataset = arff.load(open('C:/Users/gener/PycharmProjects/CKD-Tech-Project/chronic_kidney_disease.arff')) # loads the dataset
#change the filepath to where yours is
raw_data = np.array(dataset['data']) # pulls the data out into a numpy array

data = raw_data[:, :-1] # takes everything except the last columns
target = raw_data[:, -1] # just the last column

imp = Imputer(missing_values='NaN', strategy='mean', axis=0) #fixes missing data by taking values from other rows and taking the average
imp.fit(data) #iirc this function takes the average
data = imp.fit_transform(data) #inserts the average into the missing spots
data, target = SMOTE().fit_sample(data, target) # oversamples the minority class (notckd)

# param_grid = {"max_depth": [3, None],
#               "max_features": range(1, 11),
#               "min_samples_split": range(2, 11),
#               "min_samples_leaf": range(1, 11),
#               "bootstrap": [True, False]}
#
#
# metrics.OptimizeClassifier(data, target, RandomForestClassifier(), param_grid)





#instantiating estimator object
rf = RandomForestClassifier(n_estimators=250)
scores = metrics.repeatedCrossValidatedScores(data, target, rf, cv=10, iterations=5)
# rf.fit
print("title/tp/tn/fp/fn/f1/precision/sensitivity/specificity/accuracy")
metrics.printAverages("Random Forest", scores)
