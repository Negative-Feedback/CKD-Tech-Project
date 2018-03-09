#liac-arff, numpy, scipy, and scikit-learn are needed to run this
import arff
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import metrics
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import brier_score_loss, precision_score, recall_score, f1_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
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
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=0.3)


def plot_calibration_curve(est, name, fig_index):
    

    """Plot calibration curve for est w/o and with calibration. """
    # Calibrated with isotonic calibration
    isotonic = CalibratedClassifierCV(est, cv=2, method='isotonic')

    # Calibrated with sigmoid calibration
    sigmoid = CalibratedClassifierCV(est, cv=2, method='sigmoid')

    # Logistic regression with no calibration as baseline
    lr = LogisticRegression(C=1., solver='lbfgs')

    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [(lr, 'Logistic'),
                      (est, name),
                      (isotonic, name + ' + Isotonic'),
                      (sigmoid, name + ' + Sigmoid')]:
        clf.fit(data_train, target_train)
        y_pred = clf.predict(data_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(data_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(data_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        clf_score = brier_score_loss(target_test, prob_pos, pos_label = 0)
        print("%s:" % name)
        print("\tBrier: %1.3f" % (clf_score))
        print("\tPrecision: %1.3f" % precision_score(target_test, y_pred))
        print("\tRecall: %1.3f" % recall_score(target_test, y_pred))
        print("\tF1: %1.3f\n" % f1_score(target_test, y_pred))


        fraction_of_positives, mean_predicted_value = \
            calibration_curve(target_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s (%1.3f)" % (name, clf_score))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()

    # Plot calibration curve for Gaussian Naive Bayes
    #plot_calibration_curve(GaussianNB(), "Naive Bayes", 1)

    # Plot calibration curve for Linear SVC
plot_calibration_curve(svm.SVC(), "SVC", 1)

plt.show()


'''C = [1, 10, 100, 1000]
decision_function_shape = ['ovo', 'ovr']
clf = svm.SVC()
kernel = ['linear']

clf = svm.SVC(C = 1, kernel='linear', decision_function_shape='ovo') # sets up the svm

scores = metrics.repeatedCrossValidatedScores(data, target, clf, cv=10, iterations=10)
print(scores)

#param_grid = {'C': C, 'decision_function_shape': decision_function_shape, 'kernel' : kernel}
#metrics.OptimizeClassifier(data, target, clf, param_grid)
#results = cross_val_score(clf, data, target, cv = 10)
#print("Accuracy: %0.2f (+/- %0.2f)" % (results.mean()*100, results.std() * 200))'''


