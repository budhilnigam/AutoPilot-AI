"""
test_deepseek.py — Quick connectivity test for DeepSeek v3 via AWS Bedrock.
Run with: python test_deepseek.py
"""

import boto3
import json

AWS_REGION = "us-east-1"
AWS_ACCESS_KEY_ID ="ExampleAccessKey" #"AKIARFRAMVTLJFRO6PUW"
AWS_SECRET_ACCESS_KEY ="ExampleAccessKey" #"nR8QwgMK83quzIqpaIq04ogPMBLEMXoWmzDpVI8Z"
MODEL_ID = "openai.gpt-oss-20b-1:0"

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
    # Prepare request body for Invoke API (model-specific format)
    request_body = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": "Reply with exactly: DEEPSEEK_OK"
            }
        ],
        "max_tokens": 20,
        "temperature": 0
    })
    
    # Use invoke_model instead of converse
    response = client.invoke_model(
        modelId=MODEL_ID,
        body=request_body
    )
    
    # Parse the response body
    response_body = json.loads(response['body'].read())
    text = response_body['choices'][0]['message']['content']
    print(f"SUCCESS — Response: {text}")
except Exception as e:
    print(f"FAILED — {type(e).__name__}: {e}")
