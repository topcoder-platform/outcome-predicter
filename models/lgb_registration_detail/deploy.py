import sagemaker as sage
from sagemaker.predictor import csv_serializer

from config import ROLE_NAME, TRAIN_DATA_DIRECTORY, BUCKET_PREFIX, TRAINING_CORE_TYPE, DEPLOYMENT_CORE_TYPE, IMAGE_NAME

# Create session
sess = sage.Session()

# data name, account and region
data_location = sess.upload_data(TRAIN_DATA_DIRECTORY, key_prefix=BUCKET_PREFIX)
account = sess.boto_session.client('sts').get_caller_identity()['Account']
region = sess.boto_session.region_name

# image name
image = '{}.dkr.ecr.{}.amazonaws.com/{}:latest'.format(account, region, IMAGE_NAME)

# create estimator
tree = sage.estimator.Estimator(image, ROLE_NAME, 1, TRAINING_CORE_TYPE,
                                output_path="s3://{}/output".format(sess.default_bucket()),
                                sagemaker_session=sess)

# fit model
tree.fit(data_location)

# deploy model
predictor = tree.deploy(1, DEPLOYMENT_CORE_TYPE, serializer=csv_serializer)
print('Deployed on endpoint {}'.format(predictor.endpoint))
