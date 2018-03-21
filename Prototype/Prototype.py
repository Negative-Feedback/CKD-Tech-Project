import os
from flask import Flask, redirect, url_for, render_template, request
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '/Prototype'
ALLOWED_EXTENSIONS = set(['arff'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def Main_file():
    return render_template('Main.html')



@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        return 'file uploaded successfully'


if __name__ == '__main__':
    app.run(debug=True)