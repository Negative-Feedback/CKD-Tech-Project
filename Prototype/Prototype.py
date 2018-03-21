import os
from flask import Flask, redirect, url_for, render_template, request
from werkzeug.utils import secure_filename
from sklearn.externals import joblib
import arff
import numpy as np

UPLOAD_FOLDER = '/Prototype'
#ALLOWED_EXTENSIONS = set(['arff'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def Main_file():
    return render_template('Main.html')



@app.route('/results', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        dataset = arff.load(f)
        data = np.array(dataset['data'])
        predection = getPredection(data)
        predection = 0
        return render_template('Results.html', result = predection)



def getPredection(data):
    MLA = joblib.load('SVM/SVM.pkl')
    result = MLA.predict(data)
    return(result)

if __name__ == '__main__':
    app.run(debug=True)