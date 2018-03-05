#liac-arff, numpy, scipy, and scikit-learn are needed to run this
import arff
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import Imputer
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

def findMin(accuracy): # finds the smallest value in an array
    min = accuracy[0]
    for x in range (0, len(accuracy)):
        if accuracy[x] < min:
            min = accuracy[x]

    return min


def findMax(accuracy): # finds the largest value in an array
    max = accuracy[0]
    for x in range (0, len(accuracy)):
        if accuracy[x] >= max:
            max = accuracy[x]

    return max

dataset = arff.load(open('C:/Users/Matthew/PycharmProjects/CKD-Tech-Project/chronic_kidney_disease.arff')) # loads the dataset
#change the filepath to where yours is

raw_data = np.array(dataset['data']) # pulls the data out into a numpy array

data = raw_data[:, :-1] # takes everything except the last column
target = raw_data[:, -1] # just the last column



imp = Imputer(missing_values='NaN', strategy='mean', axis=0) #fixes missing data by taking values from other rows and taking the average
imp.fit(data) #iirc this fucntion takes the average
data = imp.fit_transform(data) #inserts the average into the missing spots
data, target = SMOTE().fit_sample(data, target) # oversamples the minority class (notckd)


#clf = svm.SVC(C = 1, kernel='linear', decision_function_shape='ovo') # sets up the svm
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
print("test")
clf.fit(data, target)
print(sorted(clf.cv_results_.keys()))
#results = cross_val_score(clf, data, target, cv = 10)
#print("Accuracy: %0.2f (+/- %0.2f)" % (results.mean()*100, results.std() * 200))

'''
total = 0 # counter to hold the results of all the runs for calculating an average
run = 100 # this makes it so that i can adjust the how man times the loop runs with out manually changing what temp is divided by

#this for loop is used to get an average accuracy
#We do this because our results change based on how the data is split
runResults = np.zeros(shape = (run,1)) # create an array of zeros
temp = 0 # holds accuracy score
for x in range(0, run):
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.3) # 70:30 train:test data split
    clf = svm.SVC(C = 1, kernel='rbf', decision_function_shape='ovo') # sets up the svm
    clf.fit(data_train, target_train) # trains svm
    predicted = clf.predict(data_test) # testing the svm
    temp = accuracy_score(target_test, predicted) *100 # our accuracy value
    runResults[x] = temp # accuracy value added to an array
    total += temp # accuracy value added to total
    print("Iteration" + str(x))


average = total/run # calculates  the average

min = findMin(runResults) # finds max accuracy
max = findMax(runResults) # fins min accuracy

# rounds average, min, max to 4 decimal places
min = np.round(min, 4)
max = np.round(max, 4)
average = round(average, 4)

accuracy = [average, min, max] # array of values to graph
labels = ["Average: "+ str(average), "Min: " + str(float(min)), "Max: " + str(float(max))] # labels for each bar of graph


clf = svm.SVC()
clf.fit(data_train, target_train)
predicted = clf.predict(data_test)
print(accuracy_score(target_test, predicted))
index = np.arange(3) # sets spacing on x axis
width = 0.5 # bar width
plt.bar(index, accuracy, width, align = 'center') # creates the graph
plt.xticks(index, labels) # adds labels on x axis
plt.ylabel('% Accuracy') #label for y axis
plt.title(str(run) + " Runs") # adds title to graph indicating how many iterations of the loop were run

plt.show() # displays the graph
'''
