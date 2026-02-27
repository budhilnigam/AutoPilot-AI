"""
Notification Service

Responsible for delivering alerts to external channels:
- AWS SNS (Email/SMS)
- (Future) Slack/Teams webhooks
"""

import logging
import os
import json
from typing import Dict, List, Optional, Any
import boto3
from botocore.exceptions import ClientError

from models.agent_protocol import Insight, Severity

logger = logging.getLogger(__name__)


class NotificationService:
    """Service for delivering alerts and insights"""
    
    def __init__(self, region_name: str = None, sns_topic_arn: str = None):
        """
        Initialize Notification Service.
        
        Args:
            region_name: AWS region
            sns_topic_arn: SNS Topic ARN for alerts
        """
        self.region_name = region_name or os.getenv('AWS_REGION', 'ap-south-1')
        self.sns_topic_arn = sns_topic_arn or os.getenv('SNS_TOPIC_ARN')
        
        try:
            self.sns = boto3.client('sns', region_name=self.region_name)
            logger.info("Notification Service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Notification Service: {e}")
            self.sns = None
    
    def send_alert(self, insight: Insight) -> bool:
        """
        Send an alert for a critical insight.
        
        Args:
            insight: The insight to alert on
            
        Returns:
            True if sent successfully
        """
        if not self.sns or not self.sns_topic_arn:
            logger.warning("SNS not configured, skipping alert delivery")
            # Log to console as fallback
            logger.info(f"ALERT: [{insight.severity.value}] {insight.summary}")
            return False
            
        try:
            # Format message
            subject = f"AutoPilot Alert: [{insight.severity.value}] {insight.summary[:50]}..."
            
            message = f"""AUTO-PILOT AI ALERT
--------------------------------------------------
SEVERITY: {insight.severity.value}
SUMMARY: {insight.summary}

BUSINESS IMPACT:
{insight.business_impact}

RECOMMENDATIONS:
{chr(10).join(['- ' + r for r in insight.recommendations])}

COST IMPACT: ₹{insight.cost_impact_inr:.2f} INR
CONFIDENCE: {insight.confidence_score:.0%}
--------------------------------------------------
"""
            
            self.sns.publish(
                TopicArn=self.sns_topic_arn,
                Subject=subject[:100],  # AWS limit
                Message=message
            )
            
            logger.info(f"Sent SNS alert: {subject}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to send SNS alert: {e}")
            return False
    
    def send_batch_report(self, insights: List[Insight]) -> bool:
        """
        Send a summary report of multiple insights.
        
        Args:
            insights: List of insights
            
        Returns:
            True if sent successfully
        """
        if not insights:
            return False
            
        # Filter for high priority items
        critical_insights = [i for i in insights if i.severity in [Severity.CRITICAL, Severity.HIGH]]
        
        if not critical_insights:
            return False
            
        if not self.sns or not self.sns_topic_arn:
            logger.info(f"Batch Report: {len(critical_insights)} critical issues found")
            return False
            
        try:
            subject = f"AutoPilot Report: {len(critical_insights)} Critical Issues Detected"
            
            message = "AUTO-PILOT AI BATCH REPORT\n\n"
            
            for i, insight in enumerate(critical_insights, 1):
                message += f"{i}. [{insight.severity.value}] {insight.summary}\n"
                message += f"   Impact: {insight.business_impact}\n"
                message += f"   Action: {insight.recommendations[0] if insight.recommendations else 'None'}\n\n"
            
            self.sns.publish(
                TopicArn=self.sns_topic_arn,
                Subject=subject[:100],
                Message=message
            )
            
            logger.info(f"Sent SNS batch report: {subject}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to send SNS batch report: {e}")
            return False
