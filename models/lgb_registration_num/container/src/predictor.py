#!/usr/bin/env python3

# This is the file that implements a flask server to do inferences.

from __future__ import print_function

import sys
import os
import json
import pickle
import signal
import traceback
import flask
import pandas as pd
from io import StringIO

from config_and_helper import *

prefix = '/opt/ml/'
output_path = os.path.join(prefix, 'output')
model_path = os.path.join(prefix, 'model')

# The flask app for serving predictions
app = flask.Flask(__name__)

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.
class ScoringService(object):
    model = None                # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            cls.model = joblib.load(os.path.join(model_path, 'lgb'))
        return cls.model

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.

        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        print("Preprocessing data...")
        # retrieve cache files from model path
        copy_files(model_path, 'cache', ['extended_columns.pkl', 'num_date_columns.pkl'])
        test_data_preprocessing(input).to_csv('cache/preprocessed_test_data.csv', index=False)
        test_data = pd.read_csv('cache/preprocessed_test_data.csv')

        print("Start to predict...")
        result = StringIO()
        output = pd.DataFrame()
        output['Success'] = cls.get_model().predict(test_data)
        output['Challenge ID'] = input['Challenge ID']
        output.to_csv(result, index=False)

        return result

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None

    # Convert from CSV to pandas
    if flask.request.content_type == 'text/csv':
        data = flask.request.data.decode('utf-8')
        s = StringIO(data)
        data = pd.read_csv(s)
    else:
        return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')

    print('Invoked with {} records'.format(data.shape[0]))

    # Do the prediction
    predictions = ScoringService.predict(data).getvalue()
    
    return flask.Response(response=predictions, status=200, mimetype='text/csv')
