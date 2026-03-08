"""
test_iam_user.py — Check which IAM user/identity is associated with the AWS credentials.
Run with: python test_iam_user.py
"""

import boto3

AWS_REGION = "us-east-1"
AWS_ACCESS_KEY_ID = "ExampleAccessKey" #"AKIARFRAMVTLJFRO6PUW"
AWS_SECRET_ACCESS_KEY = "ExampleAccesskey"

sts = boto3.client(
    "sts",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

try:
    identity = sts.get_caller_identity()
    print("AWS Caller Identity")
    print("-" * 40)
    print(f"Account : {identity['Account']}")
    print(f"ARN     : {identity['Arn']}")
    print(f"User ID : {identity['UserId']}")
except Exception as e:
    print(f"FAILED — {type(e).__name__}: {e}")
