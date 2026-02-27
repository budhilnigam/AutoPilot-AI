"""
AWS Billing Client

Retrieves AWS cost and usage data.
Converts USD to INR for cost-aware recommendations.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class BillingClient:
    """Client for AWS Cost Explorer and billing data"""
    
    def __init__(
        self,
        region_name: str = 'us-east-1',  # CE is only in us-east-1
        usd_to_inr_rate: float = None,
    ):
        """
        Initialize Billing client.
        
        Args:
            region_name: AWS region (Cost Explorer only in us-east-1)
            usd_to_inr_rate: USD to INR conversion rate (defaults to env or 83.0)
        """
        self.region_name = region_name
        self.usd_to_inr_rate = usd_to_inr_rate or float(
            os.getenv('USD_TO_INR_RATE', '83.0')
        )
        
        config = Config(
            region_name=self.region_name,
            retries={'max_attempts': 3, 'mode': 'adaptive'}
        )
        
        try:
            self.ce_client = boto3.client('ce', config=config)
            logger.info(
                f"Billing client initialized (USD→INR rate: {self.usd_to_inr_rate})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Billing client: {e}")
            raise
    
    def get_cost_and_usage(
        self,
        start_date: str,
        end_date: str,
        metrics: List[str] = None,
        granularity: str = 'DAILY',
        group_by: List[Dict[str, str]] = None,
    ) -> Dict:
        """
        Get cost and usage data.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            metrics: Metrics to retrieve (default: ['UnblendedCost'])
            granularity: DAILY, MONTHLY, or HOURLY
            group_by: Group by dimensions/tags
            
        Returns:
            Cost and usage data
        """
        if metrics is None:
            metrics = ['UnblendedCost']
        
        try:
            params = {
                'TimePeriod': {
                    'Start': start_date,
                    'End': end_date,
                },
                'Granularity': granularity,
                'Metrics': metrics,
            }
            
            if group_by:
                params['GroupBy'] = group_by
            
            response = self.ce_client.get_cost_and_usage(**params)
            logger.info(f"Retrieved cost data from {start_date} to {end_date}")
            return response
            
        except ClientError as e:
            logger.error(f"Failed to retrieve cost data: {e}")
            raise
    
    def get_monthly_cost_inr(self, months_back: int = 1) -> float:
        """
        Get total cost for the last N months in INR.
        
        Args:
            months_back: Number of months to look back
            
        Returns:
            Total cost in INR
        """
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=30 * months_back)
        
        response = self.get_cost_and_usage(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            metrics=['UnblendedCost'],
            granularity='MONTHLY'
        )
        
        total_usd = 0.0
        for result in response.get('ResultsByTime', []):
            amount = float(result['Total']['UnblendedCost']['Amount'])
            total_usd += amount
        
        total_inr = total_usd * self.usd_to_inr_rate
        logger.info(f"Monthly cost: ${total_usd:.2f} USD = ₹{total_inr:.2f} INR")
        return total_inr
    
    def get_service_costs_inr(
        self,
        start_date: str,
        end_date: str
    ) -> Dict[str, float]:
        """
        Get cost breakdown by AWS service in INR.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dict mapping service name to cost in INR
        """
        response = self.get_cost_and_usage(
            start_date=start_date,
            end_date=end_date,
            metrics=['UnblendedCost'],
            granularity='MONTHLY',
            group_by=[{'Type': 'DIMENSION', 'Key': 'SERVICE'}]
        )
        
        service_costs = {}
        for result in response.get('ResultsByTime', []):
            for group in result.get('Groups', []):
                service = group['Keys'][0]
                amount_usd = float(group['Metrics']['UnblendedCost']['Amount'])
                amount_inr = amount_usd * self.usd_to_inr_rate
                
                if service in service_costs:
                    service_costs[service] += amount_inr
                else:
                    service_costs[service] = amount_inr
        
        return service_costs
    
    def usd_to_inr(self, amount_usd: float) -> float:
        """Convert USD to INR"""
        return amount_usd * self.usd_to_inr_rate
    
    def forecast_cost_inr(self, months: int = 3) -> Dict[str, float]:
        """
        Get cost forecast for next N months in INR.
        
        Args:
            months: Number of months to forecast
            
        Returns:
            Dict with forecasted costs
        """
        try:
            start_date = datetime.utcnow().date()
            end_date = start_date + timedelta(days=30 * months)
            
            response = self.ce_client.get_cost_forecast(
                TimePeriod={
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d'),
                },
                Metric='UNBLENDED_COST',
                Granularity='MONTHLY'
            )
            
            total_usd = float(response['Total']['Amount'])
            mean_value = float(response.get('ForecastResultsByTime', [{}])[0]
                              .get('MeanValue', total_usd))
            
            return {
                'total_inr': total_usd * self.usd_to_inr_rate,
                'mean_monthly_inr': mean_value * self.usd_to_inr_rate,
                'forecast_months': months,
            }
            
        except ClientError as e:
            logger.error(f"Failed to get cost forecast: {e}")
            return {'total_inr': 0.0, 'mean_monthly_inr': 0.0, 'forecast_months': months}
