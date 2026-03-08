"""
Amazon CloudWatch Client

Wrapper for CloudWatch metrics and logs.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from models.core_models import MetricData, MetricType

logger = logging.getLogger(__name__)


class CloudWatchClient:
    """Client for retrieving CloudWatch metrics and logs"""
    
    def __init__(self, region_name: str = None):
        """
        Initialize CloudWatch client.
        
        Args:
            region_name: AWS region (defaults to env AWS_REGION)
        """
        self.region_name = region_name or os.getenv('AWS_REGION', 'us-east-1')
        
        config = Config(
            region_name=self.region_name,
            retries={'max_attempts': 3, 'mode': 'adaptive'}
        )
        
        try:
            self.cloudwatch = boto3.client('cloudwatch', config=config)
            self.logs = boto3.client('logs', config=config)
            logger.info(f"CloudWatch client initialized for region {self.region_name}")
        except Exception as e:
            logger.error(f"Failed to initialize CloudWatch client: {e}")
            raise
    
    def get_metric_statistics(
        self,
        namespace: str,
        metric_name: str,
        dimensions: List[Dict[str, str]],
        start_time: datetime,
        end_time: datetime,
        period: int = 300,
        statistics: List[str] = None,
    ) -> List[MetricData]:
        """
        Retrieve metric statistics from CloudWatch.
        
        Args:
            namespace: CloudWatch namespace (e.g., 'AWS/EC2')
            metric_name: Metric name
            dimensions: Metric dimensions
            start_time: Start of time range
            end_time: End of time range
            period: Period in seconds (default 5 minutes)
            statistics: Statistics to retrieve (default: ['Average'])
            
        Returns:
            List of MetricData objects
        """
        if statistics is None:
            statistics = ['Average']
        
        try:
            response = self.cloudwatch.get_metric_statistics(
                Namespace=namespace,
                MetricName=metric_name,
                Dimensions=dimensions,
                StartTime=start_time,
                EndTime=end_time,
                Period=period,
                Statistics=statistics
            )
            
            # Convert to MetricData objects
            metrics = []
            for datapoint in response.get('Datapoints', []):
                # Determine metric type from name
                metric_type = self._infer_metric_type(metric_name)
                
                for stat in statistics:
                    if stat in datapoint:
                        metric = MetricData(
                            metric_name=f"{metric_name}_{stat}",
                            metric_type=metric_type,
                            value=datapoint[stat],
                            unit=datapoint.get('Unit', 'None'),
                            timestamp=datapoint['Timestamp'].isoformat(),
                            dimensions={d['Name']: d['Value'] for d in dimensions},
                            source='cloudwatch'
                        )
                        metrics.append(metric)
            
            logger.info(f"Retrieved {len(metrics)} metric datapoints for {metric_name}")
            return metrics
            
        except ClientError as e:
            logger.error(f"Failed to retrieve CloudWatch metrics: {e}")
            raise
    
    def get_recent_metrics(
        self,
        namespace: str,
        metric_name: str,
        dimensions: List[Dict[str, str]],
        hours: int = 1,
        period: int = 300,
    ) -> List[MetricData]:
        """
        Get recent metrics for the last N hours.
        
        Args:
            namespace: CloudWatch namespace
            metric_name: Metric name
            dimensions: Metric dimensions
            hours: Number of hours to look back
            period: Period in seconds
            
        Returns:
            List of MetricData objects
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        return self.get_metric_statistics(
            namespace=namespace,
            metric_name=metric_name,
            dimensions=dimensions,
            start_time=start_time,
            end_time=end_time,
            period=period
        )
    
    def query_logs(
        self,
        log_group_name: str,
        query_string: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Query CloudWatch Logs using Insights.
        
        Args:
            log_group_name: Log group name
            query_string: CloudWatch Insights query
            start_time: Start time
            end_time: End time
            limit: Max results
            
        Returns:
            Query results
        """
        try:
            # Start query
            response = self.logs.start_query(
                logGroupName=log_group_name,
                startTime=int(start_time.timestamp()),
                endTime=int(end_time.timestamp()),
                queryString=query_string,
                limit=limit
            )
            
            query_id = response['queryId']
            
            # Poll for results
            import time
            max_wait = 30  # seconds
            waited = 0
            
            while waited < max_wait:
                result = self.logs.get_query_results(queryId=query_id)
                status = result['status']
                
                if status == 'Complete':
                    logger.info(f"Log query completed with {len(result['results'])} results")
                    return result['results']
                elif status in ['Failed', 'Cancelled']:
                    logger.error(f"Log query {status}")
                    return []
                
                time.sleep(1)
                waited += 1
            
            logger.warning("Log query timed out")
            return []
            
        except ClientError as e:
            logger.error(f"Failed to query CloudWatch Logs: {e}")
            raise
    
    def _infer_metric_type(self, metric_name: str) -> MetricType:
        """Infer metric type from metric name"""
        metric_name_lower = metric_name.lower()
        
        if 'cpu' in metric_name_lower:
            return MetricType.CPU
        elif 'memory' in metric_name_lower or 'mem' in metric_name_lower:
            return MetricType.MEMORY
        elif 'latency' in metric_name_lower or 'duration' in metric_name_lower:
            return MetricType.LATENCY
        elif 'throughput' in metric_name_lower or 'requests' in metric_name_lower:
            return MetricType.THROUGHPUT
        elif 'error' in metric_name_lower:
            return MetricType.ERROR_RATE
        elif 'disk' in metric_name_lower or 'io' in metric_name_lower:
            return MetricType.DISK_IO
        elif 'network' in metric_name_lower:
            return MetricType.NETWORK
        else:
            return MetricType.CUSTOM
