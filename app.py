from flask import Flask, render_template, request
from joblib import load
import pandas as pd
import numpy as np


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('original.html')


@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        cp = float(request.form['cp'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        fbs= float(request.form['fbs'])
        restecg = float(request.form['restecg'])
        thalach = float(request.form['thalach'])
        exang = float(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = float(request.form['slope'])
        ca = float(request.form['ca'])
        thal = float(request.form['thal'])

        pred_args = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
        # pred_args = [1.63030976,0.81649658,0.97386259,-0.07975584,1.65153615,-0.32084447,1.18326286,-2.15983218,-0.58248237,1.00523959,0.69020139,2.68437746,-0.83288091]

        ml_model = load('model/hd_svm.joblib')
        model_predcition = ml_model.predict([pred_args])
        if model_predcition == 1:
            res = 'Affected'
        else:
            res = 'Not affected'
        #return res
    return render_template('predict.html', prediction = res)

if __name__ == '__main__':
    app.run()
