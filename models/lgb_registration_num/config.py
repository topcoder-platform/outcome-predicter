# config file

# IAM role name
ROLE_NAME = 'sagemaker'
# ecr image name
IMAGE_NAME = 'my-image'
# test data path
TEST_DATA_PATH = 'test_data/test.csv'
# training data path
TRAIN_DATA_DIRECTORY = 'training_data'
# s3 bucket prefix
BUCKET_PREFIX = 'my-bucket'
# type of training core
TRAINING_CORE_TYPE = 'ml.c4.2xlarge'
# type of deployment core
DEPLOYMENT_CORE_TYPE = 'ml.t2.medium'
# output file to save to when invoking endpoint
OUT_FILE = 'output/test.csv'