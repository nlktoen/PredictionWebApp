# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 18:58:46 2020

@author: ADMIN
"""
from flask import Flask, request, url_for, redirect, render_template, jsonify
import flask
import pickle
import numpy as np
import pandas as pd

app = flask.Flask(__name__, template_folder='C:/Kat/Tuyen/Project')
model = pickle.load(open("C:/Kat/Tuyen/Project/model.pkl","rb"))
scaler= pickle.load(open("C:/Kat/Tuyen/Project/scaler.pkl","rb"))

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict',methods=['Post'])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features)
    data_unseen = pd.DataFrame([final])
    data_unseen_scaled = scaler.transform(data_unseen)
    prediction = model.predict(data_unseen_scaled)
    cl=max(max((model.predict_proba(scaler.transform([[1,2,3,4,5,6,7,8,9,10,100]])))))*100
    return render_template('index.html',prediction_text='Prediction will be: {}. Confidence level: {}%'.format(prediction,cl))

if __name__ == "__main__":
    app.run(debug=True)
