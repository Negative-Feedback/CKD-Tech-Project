import arff
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn import metrics
#from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

dataset = arff.load(open('C:/Users/gener/PycharmProjects/CKD-Tech-Project/chronic_kidney_disease.arff')) # loads the dataset
#change the filepath to where yours is
raw_data = np.array(dataset['data']) # pulls the data out into a numpy array

data = raw_data[:, :-1] # takes everything except the last column
target = raw_data[:, -1] # just the last column

imp = Imputer(missing_values='NaN', strategy='mean', axis=0) #fixes missing data by taking values from other rows and taking the average
imp.fit(data) #iirc this function takes the average
data = imp.fit_transform(data) #inserts the average into the missing spots

data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.3) #breaks the dataset into test and training data
#30% of data is test data

from sklearn.datasets import load_iris

iris = load_iris()
type(iris)
# store feature matrix in "X"
X = iris.data
# store response vector in "y"
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.4, random_state = 4)

# try K=1 through K=25 and record testing accuracy
k_range = range(1,26)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
plt.show()