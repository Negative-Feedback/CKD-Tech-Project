import os
from flask import Flask, redirect, url_for, render_template, request
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '/Prototype'
#ALLOWED_EXTENSIONS = set(['arff'])

app = Flask(__name__)
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def Main_file():
    return render_template('Main.html')



@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        data = request.get_data()
        data = str(data)
        data = data[2:-1]
        fo = open("temp_upload.arff", "w")
        lines_of_text = ["@relation Chronic_Kidney_Disease", "\n", "\n", "@attribute 'sg' {1.005,1.010,1.015,1.020,1.025}", "\n" ,"@attribute 'al' {0,1,2,3,4,5}", "\n" ,"@attribute 'rbc' {normal,abnormal}", "\n" ,"@attribute 'pc' {normal,abnormal}", "\n" ,"@attribute 'hemo' numeric", "\n" ,"@attribute 'pcv' numeric", "\n" ,"@attribute 'htn' {yes,no}", "\n" ,"@attribute 'dm' {yes,no}" ,"\n", "\n" ,"@data", "\n", data]
        fo.writelines(lines_of_text)
        fo.close()
        return 'file uploaded successfully'


if __name__ == '__main__':
    app.run(debug=True)