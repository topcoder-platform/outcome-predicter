#!/usr/bin/env bash
set -e

SAGEMAKER_ROLE_NAME=$1

if [[ -z "${SAGEMAKER_ROLE_NAME}" ]]; then
  >&2 echo "Error: Role name missed"
  >&2 echo ""
  >&2 echo "Usage: $0 <SAGEMAKER_ROLE_NAME>"
  exit 1
fi

>&2 echo 'Creating SageMaker role...'
ASSUME_ROLE_POLICY=$(
  cat <<EOM
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {
                "Service": "sagemaker.amazonaws.com"
            },
            "Action": "sts:AssumeRole"
        }
    ]
}
EOM
)

SAGEMAKER_FULL_ACCESS_POLICY='arn:aws:iam::aws:policy/AmazonSageMakerFullAccess'

roleArn=$(aws iam create-role --role-name "${SAGEMAKER_ROLE_NAME}" --assume-role-policy "${ASSUME_ROLE_POLICY}" --output text --query Role.Arn)
>&2 echo "Role ARN: ${roleArn}"

>&2 echo "Attaching policies..."
aws iam attach-role-policy --role-name "${SAGEMAKER_ROLE_NAME}" --policy-arn "${SAGEMAKER_FULL_ACCESS_POLICY}"
>&2 echo "Attached policies."

>&2 echo "The IAM role ${SAGEMAKER_ROLE_NAME} for SageMaker is ready"
