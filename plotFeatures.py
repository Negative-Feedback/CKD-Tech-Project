import arff
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import metrics
import warnings
warnings.filterwarnings("ignore")


# classification threshold

# open the arff file
dataset = arff.load(open('chronic_kidney_disease.arff'))

# pulls the data into a numpy array
raw_data = np.array(dataset['data'])

# takes everything except the last column
data = raw_data[:, :-1]

# just the last column
target = raw_data[:, -1]

# fixes missing data by taking values from other rows and taking the average
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

# this function takes the average of every column excluding the unknown values
imp.fit(data)

# inserts the average into the missing spots
data = imp.fit_transform(data)

data, target = SMOTE().fit_sample(data, target)

barHeights = metrics.UnivariateSelection(data, target)
plt.bar(range(1, 25), barHeights)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.show()