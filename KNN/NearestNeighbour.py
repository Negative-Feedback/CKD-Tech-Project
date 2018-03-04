import arff
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.metrics import accuracy_score
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

#instantiating estimator object
knn = KNeighborsClassifier(n_neighbors=1)

#fit the model with data
knn.fit(X,y)

#predict the response for new observations
#returns a NumPy array of predictions
X_New = [[3,5,4,2], [5,4,3,2]]
print(knn.predict(X_New))

#instantiate the model (using the value K=5)
knn = KNeighborsClassifier(n_neighbors=5)

# fit the model with data
knn.fit(X,y)

#predict the response for new observations
print(knn.predict(X_New))

logreg = LogisticRegression()

#fit the model with data
logreg.fit(X,y)

print(logreg.predict(X_New))