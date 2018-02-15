import arff
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression

dataset = arff.load(open('C:/Users/Tyler/PycharmProjects/CKD-Tech-Project/chronic_kidney_disease.arff')) # loads the dataset
#change the filepath to where yours is
raw_data = np.array(dataset['data']) # pulls the data out into a numpy array

data = raw_data[:, :-1] # takes everything except the last column
target = raw_data[:, -1] # just the last column

imp = Imputer(missing_values='NaN', strategy='mean', axis=0) #fixes missing data by taking values from other rows and taking the average
imp.fit(data) #iirc this fucntion takes the average
data = imp.fit_transform(data) #inserts the average into the missing spots
#x                           y
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.3) #breaks the dataset into test and training data
#30% of data is test data

model = LogisticRegression() #Creates a copy of te function LogisticRegression and names it as model
model = model.fit(data_train, target_train) #Fits the data of data_train and target_train on the model
# check the accuracy on the training set using score
temp = model.score(data_train, target_train)
print(str(temp * 100) + '%')

# check the accuracy on the training set using 
#model.fit(data_train, target_train)
prediction = model.predict(data_test)
accuracy = 0.
for n in range(target_test.size):
    if target_test[n] == prediction[n]:
        accuracy += 1.
accuracy /= target_test.size
print("The predictions were " + str(accuracy * 100.) + "% accurate")