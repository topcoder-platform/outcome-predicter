import boto3

import pandas as pd
import numpy as np
import csv

from io import StringIO

from config import IMAGE_NAME, TEST_DATA_PATH, OUT_FILE

# create client and get endpoint name
client = boto3.client('sagemaker')
endpoint_name = client.list_endpoints(SortBy='CreationTime', SortOrder='Descending',
                                      NameContains=IMAGE_NAME)['Endpoints'][0]['EndpointName']
runtime_client = boto3.client('runtime.sagemaker')

# read csv
out = StringIO()
pd.read_csv(TEST_DATA_PATH, ';').to_csv(out)

# invoke endpoint
response = runtime_client.invoke_endpoint(EndpointName = endpoint_name,
                                          ContentType = 'text/csv',
                                          Body = out.getvalue())
# save result as csv
buffer = StringIO(response['Body'].read().decode('ascii'))
pd.read_csv(buffer).to_csv(OUT_FILE)
print('Saved output in file {}'.format(OUT_FILE))