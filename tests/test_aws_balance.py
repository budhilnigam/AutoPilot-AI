"""
test_aws_balance.py — Check the current monthly AWS bill/balance using Cost Explorer.

Run with: python tests/test_aws_balance.py
"""

import os
from datetime import date, timedelta
from pathlib import Path

import boto3
from dotenv import load_dotenv

# Load credentials from .env in the root directory
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

def check_aws_balance():
    print("Checking AWS Cost Explorer for current month's estimated spend...")
    print("-" * 60)
    
    # Cost Explorer endpoint is globally in us-east-1
    try:
        client = boto3.client(
            "ce",
            region_name="us-east-1",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
    except Exception as e:
        print(f"FAILED to initialize boto3 client: {e}")
        return

    # Set up date range for the current month
    today = date.today()
    first_day_of_month = today.replace(day=1)
    
    # End date must be > Start date. If today is the 1st, make End the 2nd
    # Otherwise make End today (Cost Explorer End date is exclusive)
    end_date = today if today > first_day_of_month else today + timedelta(days=1)
    
    try:
        response = client.get_cost_and_usage(
            TimePeriod={
                'Start': first_day_of_month.strftime('%Y-%m-%d'),
                'End': end_date.strftime('%Y-%m-%d')
            },
            Granularity='MONTHLY',
            Metrics=['UnblendedCost']
        )
        
        # Parse the result
        result = response['ResultsByTime'][0]
        cost = result['Total']['UnblendedCost']
        amount = float(cost['Amount'])
        unit = cost['Unit']
        
        print(f"[OK] Authentication successful!")
        print(f"[OK] Billing span: {result['TimePeriod']['Start']} to {result['TimePeriod']['End']}")
        print(f"[OK] Current estimated spend: {amount:.2f} {unit}")
        
    except client.exceptions.DataUnavailableException as e:
        print(f"\n[ERROR] Cost data unavailable: {e}")
    except Exception as e:
        print(f"\n[ERROR] Failed to retrieve AWS balance: {e}")
        print("\nHint: Your IAM user or role likely needs the 'ce:GetCostAndUsage' permission.")
        print("Go to AWS IAM Console, attach a inline policy to the user with Action: ce:GetCostAndUsage on Resource: *")

if __name__ == "__main__":
    check_aws_balance()