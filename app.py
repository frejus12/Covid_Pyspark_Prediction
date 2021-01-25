from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from datetime import date
import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('model_path.pickle', 'rb'))
model_vector = pickle.load(open('vectorizer_path.pickle', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')
standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        date =  datetime.datetime.strptime(request.form['Date'],'%Y-%m-%d')
        etat = str(request.form['Etat'])
        cas = int(request.form['Nombre_de_cas'])
        longitude = float(request.form['Longitude'])
        latitude = float(request.form['Latitude'])
        day = date.year
        month = date.month
        year= date.year
        df = pd.DataFrame(columns=['lat', 'long', 'cases', 'AK', 'AL', 'AR', 'AZ', 'CA', 'CO',
       'CT', 'DE', 'FL', 'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA',
       'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NA', 'NC', 'ND', 'NE',
       'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD',
       'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY', 'year', 'month',
       'day'])
        df.loc[0]=[latitude,longitude,cas,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,year,month,day]
        df[etat][df[etat]==0]=1
        x = model_vector.transform(df)
        prediction=model.predict(x)
        output= round(prediction[0])
        if output<0:
            return render_template('index.html',prediction_texts="Sorry ")
        else:
            return render_template('index.html',prediction_text=" Le nombre de morts présumé est {}".format(output ))
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)