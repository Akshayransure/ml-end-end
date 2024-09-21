import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)

app = application


# import ridge regressor and scaler pickle

ridge_model = pickle.load(open('models/ridge1.pkl','rb'))
standard_scaler = pickle.load(open('models/scaler.pkl','rb'))



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == "POST":
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scale = standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])

        print("\n\nnew_data_scale : ", new_data_scale,"\n\n",ridge_model,"\n\n akshay ransyeee\n\n")
        result = ridge_model.predict(new_data_scale)

        print("\n\n\n result", result,"\n\n\n")

        return render_template('home.html',results = result[0])
    else:
        return render_template('home.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0')
