import os
import arff
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.externals import joblib
from flask import Flask, redirect, url_for, render_template, request
import metrics

UPLOAD_FOLDER = '/Prototype'
#ALLOWED_EXTENSIONS = set(['arff'])

app = Flask(__name__)
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

defaults = [1.0174079320113256, 1.0169491525423728, 0.8104838709677419, 0.7731343283582089,
            12.526436781609195, 38.88449848024316, 0.3693467336683417, 0.3442211055276382]

@app.route('/')
def Main_file():
    fo = open("temp_upload.arff", "w")
    fo.write("")
    fo.close()
    return render_template('Main.html')



@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # get data from webpage
        data = request.get_data()

        # remove unnecessary characters
        data = str(data)
        data = data[2:-1]

        # classify the data
        return classify(data)


def classify(str_data):
    # retrieve data
    raw_data = str_data.split(",")
    # get certainty estimation
    certainty = returnCertainty(raw_data)

    # process missing data
    for x in range(8):
        if raw_data[x] == "?":
            raw_data[x] = str(defaults[x])

    # transfer data into classifier-readable format
    data = np.zeros(8, float)

    # cast float data to floats
    data[0] = float(raw_data[0])
    data[1] = float(raw_data[1])

    # cast binary data to float
    if raw_data[2] == 'normal':
        data[2] = 1.
    elif raw_data[2] == 'abnormal':
        data[2] = 0.
    else:
        data[2] = float(raw_data[2])

    if raw_data[3] == 'normal':
        data[3] = 1.
    elif raw_data[3] == 'abnormal':
        data[3] = 0.
    else:
        data[3] = float(raw_data[3])

    # cast float data to floats
    data[4] = float(raw_data[4])
    data[5] = float(raw_data[5])

    # cast binary data to float
    if raw_data[6] == 'yes':
        data[6] = 1.
    elif raw_data[6] == 'no':
        data[6] = 0.
    else:
        data[6] = float(raw_data[6])
    if raw_data[7] == 'yes':
        data[7] = 1.
    elif raw_data[7] == 'no':
        data[7] = 0.
    else:
        data[7] = float(raw_data[7])

    # scale float data the same way the training data was
    data[0] = (data[0] - 1.005) / 0.02
    data[1] = (data[1]) / 5
    data[4] = (data[4] - 3.1) / 14.7
    data[5] = (data[5] - 9) / 45

    # load the classifier
    clf = joblib.load('classifier.pkl')

    # classify the patient
    prediction = clf.predict([data])[0]

    # return the result
    if prediction == '1':
        return "This person is CKD positive"
    else:
        return "This person is CKD negative"

def returnCertainty(raw_data):
    # quickly train a model to figure out how accurate it is with the given features
    data, target = metrics.preprocess(k=8, fsiter=10)

    # delete columns that weren't given
    for x in range(7, -1, -1):
        if raw_data[x] == "?":
            data = np.delete(data, x, 1)

    # use cross validation to estimate the accuracy of the classifier
    temp = metrics.repeatedCrossValidatedScores(data, target, KNeighborsClassifier(n_neighbors=1), iterations=1, cv=10)
    return np.average(temp['test_accuracy'])


if __name__ == '__main__':
    app.run(debug=True)
