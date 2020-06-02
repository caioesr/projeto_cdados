#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Caio Emmanuel
"""

from sklearn.externals import joblib
import numpy as np
import pandas as pd
from flasgger import Swagger

from flask import Flask, request
app = Flask(__name__)
swagger = Swagger(app)

final_clf = joblib.load('./final_clf.pkl')

@app.route('/detect_fraud', methods=['POST'])
def predict_digit():
    """Example endpoint returning a detection of a fraud
    ---
    parameters:
        - name: CSV File
          in: formData
          type: file
          required: true
    """
    dataset = pd.read_csv(request.files['CSV File'])
    X_test = dataset.loc[:dataset.shape[0]]
    #X_test = np.array(dataset.loc[0])
    #X_test = X_test.reshape(1,-1)
    y_pred = final_clf.predict(X_test)
    y_pred = list(y_pred)
    predictions_dict = {}
    for i in range(dataset.shape[0]):
        predictions_dict[i] = ("FRAUD" if y_pred[i] == 1 else "NON FRAUD")
    return predictions_dict

if __name__ == '__main__':
    app.run()