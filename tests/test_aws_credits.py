"""
test_aws_credits.py — Check AWS credits applied to the account using Cost Explorer.
Run with: python tests/test_aws_credits.py

Note: AWS does not have a public API to retrieve "remaining credit balance". 
This script uses Cost Explorer to sum up applied credits to your bill over the current and past month.
"""

import os
from datetime import date, timedelta
import calendar
from pathlib import Path

import boto3
from dotenv import load_dotenv

# Load credentials from root .env
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

def check_aws_credits():
    print("Checking AWS Cost Explorer for Applied Credits...")
    print("-" * 60)

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

    # Check last month and this month to see if credits were applied
    today = date.today()
    # Go back roughly 30 days
    start_date = today.replace(day=1) - timedelta(days=calendar.monthrange(today.year, today.month - 1 if today.month > 1 else 12)[1] if today.day == 1 else 30)
    start_date = start_date.replace(day=1) # start of previous month
    
    end_date = today + timedelta(days=1)

    try:
        response = client.get_cost_and_usage(
            TimePeriod={
                'Start': start_date.strftime('%Y-%m-%d'),
                'End': end_date.strftime('%Y-%m-%d')
            },
            Granularity='MONTHLY',
            Metrics=['UnblendedCost'],
            GroupBy=[
                {
                    'Type': 'DIMENSION',
                    'Key': 'RECORD_TYPE'
                }
            ]
        )

        print("[OK] Fetch successful.\n")
        credits_found = False

        for monthly_result in response['ResultsByTime']:
            month_start = monthly_result['TimePeriod']['Start']
            for group in monthly_result['Groups']:
                record_type = group['Keys'][0]
                cost = float(group['Metrics']['UnblendedCost']['Amount'])
                unit = group['Metrics']['UnblendedCost']['Unit']
                
                # Credits, Discounts, or Refunds
                if record_type in ['Credit', 'Discount', 'Refund'] and cost != 0:
                    credits_found = True
                    print(f"[{month_start}] Applied {record_type}: {cost:.2f} {unit}")

        if not credits_found:
            print("[INFO] No credits or discounts found to be applied in the checked billing period.")
            print("Note: AWS API cannot fetch 'remaining unapplied credit balance', only credits that have already offset a bill.")

    except Exception as e:
        print(f"\n[ERROR] Failed to retrieve AWS credit data: {e}")

if __name__ == "__main__":
    check_aws_credits()
