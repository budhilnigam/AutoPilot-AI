"""
test_claude.py — Quick connectivity test for Anthropic Claude via AWS Bedrock.
Tries the bare model ID first, then lists available inference profiles
so we can pick the correct one for ap-south-1.
Run with: python test_claude.py
"""

import boto3
import json

AWS_REGION = "ap-south-1"
AWS_ACCESS_KEY_ID = "AKIARFRAMVTLJFRO6PUW"
AWS_SECRET_ACCESS_KEY = "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"
MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"

runtime = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

bedrock = boto3.client(
    "bedrock",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)

# ── 1. Try bare model ID ──────────────────────────────────────────────────────
print(f"[1] Testing bare model ID: {MODEL_ID}")
print(f"    Region: {AWS_REGION}")
try:
    response = runtime.converse(
        modelId=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [{"text": "Reply with exactly: CLAUDE_OK"}],
            }
        ],
        inferenceConfig={"maxTokens": 20, "temperature": 0},
    )
    text = response["output"]["message"]["content"][0]["text"]
    print(f"    SUCCESS — {text}")
except Exception as e:
    print(f"    FAILED — {type(e).__name__}: {e}")

print()

# ── 2. List all inference profiles that contain Claude ───────────────────────
print("[2] Listing available inference profiles mentioning 'claude'...")
try:
    paginator = bedrock.get_paginator("list_inference_profiles")
    for page in paginator.paginate(typeEquals="SYSTEM_DEFINED"):
        for p in page.get("inferenceProfileSummaries", []):
            name = p.get("inferenceProfileName", "")
            pid  = p.get("inferenceProfileId", "")
            if "claude" in name.lower() or "claude" in pid.lower():
                print(f"    {pid:<60} {name}")
except Exception as e:
    print(f"    list_inference_profiles error: {type(e).__name__}: {e}")
    # Fall back to listing foundation models
    print("    Falling back to list_foundation_models...")
    try:
        resp = bedrock.list_foundation_models(byOutputModality="TEXT")
        for m in resp.get("modelSummaries", []):
            mid = m.get("modelId", "")
            if "claude" in mid.lower():
                print(f"    {mid}")
    except Exception as e2:
        print(f"    list_foundation_models error: {e2}")

print()

# ── 3. Try cross-region inference profile IDs ────────────────────────────────
candidates = [
    "ap.anthropic.claude-haiku-4-5-20251001-v1:0",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "anthropic.claude-haiku-3-5-20241022-v1:0",
    "ap.anthropic.claude-3-5-haiku-20241022-v1:0",
    "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
]

print("[3] Trying candidate model/profile IDs...")
for candidate in candidates:
    try:
        resp = runtime.converse(
            modelId=candidate,
            messages=[{"role": "user", "content": [{"text": "Say OK"}]}],
            inferenceConfig={"maxTokens": 10, "temperature": 0},
        )
        text = resp["output"]["message"]["content"][0]["text"]
        print(f"    WORKS  {candidate}  =>  '{text}'")
    except Exception as e:
        short = str(e)[:90]
        print(f"    FAILED {candidate}  =>  {short}")
