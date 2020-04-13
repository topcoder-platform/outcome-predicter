# outcome-predicter

## Structure

- **models** includes ML code that can be deployed to Amazon Sagemaker.
- **api** includes a web server that exposes an API utilizing different models to get prediction results.
- **scripts** includes shell scripts to manage AWS resources.

## Models

There are multiple models we could try and compare their performance.

### lgb_registration_num

This model uses several public data of one challenge, including number of registration.

### lgb_registration_detail

This model uses several public data of one challenge, including number of registration and registration details.

Please read the README inside the root directory of each model for how to deploy the models to Sagemaker.
A privileged IAM role is required to access Sagemaker resources during deployment
which can be created with script [create_sagemaker_role](#create_sagemaker_role).

## API
Refer `api/README.md` for how to run the API server.
You need to [deploy models](#models) first so that model configuration can be set up for the API server.

## Scripts

### create_sagemaker_role
Creates an IAM role to access Sagemaker resources.

#### Usage

``` bash
scripts/create_sagemaker_role.sh <SAGEMAKER_ROLE_NAME>
```

#### Arguments

- `SAGEMAKER_ROLE_NAME` the role name which should be unique among other role names.
