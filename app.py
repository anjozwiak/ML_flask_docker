
import numpy as np
from joblib import load
from flask import (
        Flask, 
        request, 
        jsonify
)
import pickle

with open('./model.bin', 'rb') as f_in: 
    (dv, model) = pickle.load(f_in)
    
app = Flask(__name__)

# Create an API end point
@app.route('/api/predict', methods=['GET'])
def get_prediction():

    #Age
    age = float(request.args.get('age'))
    #FrequentFlyer
    FrequentFlyer = request.args.get('FF')
    #AnnualIncomeClass
    AnnualIncomeClass = request.args.get('AI')
    #ServicesOpted
    ServicesOpted = float(request.args.get('SO'))
    #AccountSyncedToSocialMedia
    AccountSyncedToSocialMedia = request.args.get('SM')
    #BookedHotelOrNot
    BookedHotelOrNot = request.args.get('HT')

    # The features of the observation to predict
    features = [{'FrequentFlyer': FrequentFlyer,
 'AnnualIncomeClass': AnnualIncomeClass,
 'AccountSyncedToSocialMedia': AccountSyncedToSocialMedia,
 'BookedHotelOrNot': BookedHotelOrNot,
 'Age': age,
 'ServicesOpted': ServicesOpted}
               ]

    X = dv.transform(features)
    y_pred = model.predict_proba(X)[0,1]
    resuly = {
    'churn': y_pred
    }
    return jsonify(features=features,
    y_pred=y_pred)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
