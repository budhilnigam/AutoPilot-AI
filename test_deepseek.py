"""
test_deepseek.py — Quick connectivity test for DeepSeek v3 via AWS Bedrock.
Run with: python test_deepseek.py
"""

import boto3
import json

AWS_REGION = "ap-south-1"
AWS_ACCESS_KEY_ID = "AKIARFRAMVTLJFRO6PUW"
AWS_SECRET_ACCESS_KEY = "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"
MODEL_ID = "deepseek.v3-v1:0"

client = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

print(f"Testing DeepSeek v3: {MODEL_ID}")
print(f"Region: {AWS_REGION}")
print("-" * 50)

try:
    response = client.converse(
        modelId=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [{"text": "Reply with exactly: DEEPSEEK_OK"}],
            }
        ],
        inferenceConfig={"maxTokens": 20, "temperature": 0},
    )
    text = response["output"]["message"]["content"][0]["text"]
    print(f"SUCCESS — Response: {text}")
except Exception as e:
    print(f"FAILED — {type(e).__name__}: {e}")
