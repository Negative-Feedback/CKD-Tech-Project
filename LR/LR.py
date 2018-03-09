import arff
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import metrics

dataset = arff.load(open('C:/Users/Tyler/PycharmProjects/CKD-Tech-Project/chronic_kidney_disease.arff')) # loads the dataset
#change the filepath to where yours is
raw_data = np.array(dataset['data']) # pulls the data out into a numpy array

data = raw_data[:, :-1] # takes everything except the last column
target = raw_data[:, -1] # just the last column

imp = Imputer(missing_values='NaN', strategy='mean', axis=0) #fixes missing data by taking values from other rows and taking the average
imp.fit(data) #iirc this fucntion takes the average
data = imp.fit_transform(data) #inserts the average into the missing spots
data, target = SMOTE().fit_sample(data, target)

model = LogisticRegression(C=1000) #Creates a copy of te function LogisticRegression and names it as model

results = metrics.repeatedCrossValidatedScores(data, target, model, cv =10, iterations=50)#Gives avaerage accuracy
print("Accuracy: %0.2f (+/- %0.2f)" % (results['test_accuracy'].mean()*100, results['test_accuracy'].std() * 200))#prints results

