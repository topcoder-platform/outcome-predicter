# TC - Create MachineLearning workflow with AWS Sagemaker

## Structure
The files `deploy.py`, `validate.py`, and `config.py` handle deployment, validation and configuration respectively. The last simply stores config variables. Dockerfile is located in `./container`, which also contains the file `./build_and_push.sh` which will build Docker file and push it to Amazon ECR. The source code for the model is in the `./container/src` folder, which contains `train`, `serve`, and `predictor.py`, that handle model training, serving and predicting respectively.

In order to build a production grade inference server into the container, we use [nginx](http://nginx.org/) (a light-weight layer that handles the incoming HTTP requests and manages the I/O in and out of the container efficiently), [gunicorn](http://gunicorn.org/) (a WSGI pre-forking worker server that runs multiple copies of the application and load balances between them) and [flask](http://flask.pocoo.org/) (a simple web framework that lets us respond to call on the /ping and /invocations endpoints without having to write much code). The files `nginx.conf`, `wsgi.py` and `predictor.py` (all in `container/src`) handle nginx config and the flask web server.

## Prerequisites
1. Python 3.7.x
2. Docker (docker & docker-compose)
3. AWS CLI 1.x

Please configure the AWS CLI with the command `aws configure`. Then, create an IAM user role with full SageMaker permissions from AWS Console (https://console.aws.amazon.com/iam/home#/roles) and note down the role name in the config (see below). See screenshot: `docs/screenshots/iam-role.jpeg` for what the IAM console should look like afterwards.

## Config Variables

You can set a variety of config variables in the file `config.py`. The one variable you **need** to set is `ROLE_NAME`, which is the name of an AWS Role with full sagemaker permissions. **`IMAGE_NAME` must be the same as the `image-name` you use when pushing the container to ECR** (see below). The rest can be left default.

- **ROLE_NAME**: AWS IAM role name (must have full sagemaker permissions), defaults to `sagemaker`						
- **IMAGE_NAME**: Name of ecr image (see deployment section below), defaults to `my-image`
- **TEST_DATA_PATH**: Path where the test data file is located, defaults to `test_data/test.csv`
- **TRAIN_DATA_DIRECTORY**: Path of training data directory, defaults to `training_data`
- **BUCKET_PREFIX**: Prefix of S3 bucket, default to `my-bucket`								
- **TRAINING_CORE_TYPE**: The type of training core type (must be class xlarge), defaults to `ml.c4.2xlarge`
- **DEPLOYMENT_CORE_TYPE**: The deployment core type, defaults to `ml.t2.medium`
- **OUT_FILE**: The path of the file written to when validating, defaults to `output/test.csv`

### Environment variables
When you create an inference server, you can control some of Gunicorn's options via environment variables.

- **MODEL_SERVER_WORKERS**: number of workers, defaults to the number of CPU cores
- **MODEL_SERVER_TIMEOUT**: timeout, defaults to 60 seconds

## Prepare training data and test data
Before you procceed to deployment and validation you must prepare training data for deployment and test data for validation.
It can be done by first `cd` into the `./container/src` folder and then invoking the following commands:

``` bash
mkdir ../../test_data ../../training_data # create folders to accommodate data
python divide_data.py challenge-with-registration.xlsx ../../training_data/data.csv ../../test_data/test.csv
```

## Deployment

First, set config variables (see above). Make sure AWS CLI is set up. If you haven't, log into the AWS CLI with `aws configure`. Please note the config variables must be set correctly and your account must have permissions to use the cores you have selected. The `ROLE_NAME` config value must be set to the name of a role that has full SageMaker permissions.


Install requirements:

```bash
pip install -r requirements.txt
```

 Now, `cd` into the `container` folder. Create and upload container image to Amazon ECR by calling below command, where `image-name` is the name of the created image (it must be the same as the `IMAGE_NAME` set in config above).
 For windows, use the git bash instead of cmd (cmd does not support `.sh` files)

```bash
cd container
sh build_and_push.sh <image-name>
```

`cd` back into the root folder and deploy model:
```bash 
cd ../
python deploy.py
```

This will create the s3 bucket, upload the csv data, build the model, train the model, and deploy to Sagemaker.

Validate model by calling:

```bash
python validate.py
```

This will send a request to the SageMaker endpoint with the test data in the path set in config, printing the result in the console.

## Validation
After deploying code by following the steps above, go to https://s3.console.aws.amazon.com/s3/. You will see a new bucket has been created with the prefix 'sagemaker'. Click on that to see your bucket.

Go to https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/endpoints (replace 'us-east-1' with your region) to see that your endpoint has been created.

Call `validate.py` as seen above. This will call the endpoint using the test data in `./test.csv`. It will output the prediction to a csv file set in config (see above).

### Screenshots
Find screenshots in the `docs/screenshots` directory. They contain images of all the steps for deployment and validation. See `iam-role.jpeg` for where to go to create your IAM role. See `build_and_push.jpeg`, `deploy.jpeg`, `validate.jpeg` to see deployment and validation respecively. See `s3-bucket.jpeg` and `sagemaker-endpoint.jpeg` to see where the bucket and endpoint will be visible on the AWS website after deploying.
