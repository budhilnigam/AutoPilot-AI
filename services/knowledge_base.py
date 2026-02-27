"""
Knowledge Base Service

Implements RAG (Retrieval-Augmented Generation) using:
- Amazon Bedrock Knowledge Bases
- Titan Embeddings
- S3 for document storage
"""

import json
import logging
import os
import hashlib
from typing import Dict, List, Optional, Any
from datetime import datetime
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from models.core_models import Configuration, MetricData

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    Knowledge Base service for storing and retrieving infrastructure context.
    
    Provides:
    - Configuration storage in S3
    - Metric indexing for historical analysis
    - RAG-based context retrieval using Bedrock Knowledge Bases
    """
    
    def __init__(
        self,
        region_name: str = None,
        s3_bucket: str = None,
        knowledge_base_id: str = None,
    ):
        """
        Initialize Knowledge Base service.
        
        Args:
            region_name: AWS region
            s3_bucket: S3 bucket for document storage
            knowledge_base_id: Bedrock Knowledge Base ID
        """
        self.region_name = region_name or os.getenv('AWS_REGION', 'ap-south-1')
        self.s3_bucket = s3_bucket or os.getenv('S3_BUCKET_NAME')
        self.knowledge_base_id = knowledge_base_id or os.getenv('KNOWLEDGE_BASE_ID')
        
        if not self.s3_bucket:
            raise ValueError("S3 bucket not configured")
        
        config = Config(
            region_name=self.region_name,
            retries={'max_attempts': 3, 'mode': 'adaptive'}
        )
        
        try:
            self.s3 = boto3.client('s3', config=config)
            
            # Bedrock Agent Runtime for Knowledge Base queries
            if self.knowledge_base_id:
                self.bedrock_agent = boto3.client(
                    'bedrock-agent-runtime',
                    config=config
                )
            else:
                self.bedrock_agent = None
                logger.warning("Knowledge Base ID not provided - RAG disabled")
            
            logger.info(
                f"Knowledge Base initialized (Bucket: {self.s3_bucket}, "
                f"KB ID: {self.knowledge_base_id})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Knowledge Base: {e}")
            raise
    
    def store_configuration(
        self,
        config: Configuration,
        prefix: str = "configurations"
    ) -> str:
        """
        Store configuration snapshot in S3.
        
        Args:
            config: Configuration object
            prefix: S3 prefix/folder
            
        Returns:
            S3 object key
        """
        try:
            # Generate S3 key
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            filename = f"{config.config_type}_{timestamp}_{config.config_hash[:8]}.json"
            s3_key = f"{prefix}/{config.config_type}/{filename}"
            
            # Prepare document with metadata
            document = {
                'config_type': config.config_type,
                'config_content': config.config_content,
                'config_hash': config.config_hash,
                'timestamp': config.timestamp,
                'source': config.source,
                'metadata': config.metadata,
            }
            
            # Upload to S3
            self.s3.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=json.dumps(document, indent=2),
                ContentType='application/json',
                Metadata={
                    'config-type': config.config_type,
                    'timestamp': config.timestamp,
                    'source': config.source,
                }
            )
            
            logger.info(f"Stored configuration: s3://{self.s3_bucket}/{s3_key}")
            return s3_key
            
        except ClientError as e:
            logger.error(f"Failed to store configuration: {e}")
            raise
    
    def store_metrics(
        self,
        metrics: List[MetricData],
        prefix: str = "metrics"
    ) -> str:
        """
        Store metric data in S3 for historical analysis.
        
        Args:
            metrics: List of metric data points
            prefix: S3 prefix/folder
            
        Returns:
            S3 object key
        """
        if not metrics:
            logger.warning("No metrics to store")
            return ""
        
        try:
            # Group by timestamp/period
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            s3_key = f"{prefix}/{timestamp}.json"
            
            # Convert to JSON-serializable format
            metrics_data = {
                'timestamp': timestamp,
                'count': len(metrics),
                'metrics': [
                    {
                        'metric_name': m.metric_name,
                        'metric_type': m.metric_type.value,
                        'value': m.value,
                        'unit': m.unit,
                        'timestamp': m.timestamp,
                        'dimensions': m.dimensions,
                        'source': m.source,
                    }
                    for m in metrics
                ]
            }
            
            # Upload to S3
            self.s3.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=json.dumps(metrics_data, indent=2),
                ContentType='application/json'
            )
            
            logger.info(f"Stored {len(metrics)} metrics: s3://{self.s3_bucket}/{s3_key}")
            return s3_key
            
        except ClientError as e:
            logger.error(f"Failed to store metrics: {e}")
            raise
    
    def query_context(
        self,
        query: str,
        max_results: int = 5,
        min_score: float = 0.6,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context using RAG.
        
        Uses Bedrock Knowledge Bases with Titan Embeddings for semantic search.
        
        Args:
            query: Natural language query
            max_results: Maximum results to return
            min_score: Minimum similarity score threshold (0.0-1.0)
            
        Returns:
            List of relevant documents with scores
        """
        if not self.bedrock_agent or not self.knowledge_base_id:
            logger.warning("Knowledge Base RAG not available")
            return []
        
        try:
            response = self.bedrock_agent.retrieve(
                knowledgeBaseId=self.knowledge_base_id,
                retrievalQuery={'text': query},
                retrievalConfiguration={
                    'vectorSearchConfiguration': {
                        'numberOfResults': max_results
                    }
                }
            )
            
            # Filter by similarity score
            results = []
            for retrieval_result in response.get('retrievalResults', []):
                score = retrieval_result.get('score', 0.0)
                
                if score >= min_score:
                    results.append({
                        'content': retrieval_result.get('content', {}).get('text', ''),
                        'score': score,
                        'location': retrieval_result.get('location', {}),
                        'metadata': retrieval_result.get('metadata', {}),
                    })
            
            logger.info(
                f"Retrieved {len(results)} context documents "
                f"(min score: {min_score:.2f})"
            )
            return results
            
        except ClientError as e:
            logger.error(f"Failed to query Knowledge Base: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error querying Knowledge Base: {e}")
            return []
    
    def index_metrics(
        self,
        metric_name: str,
        time_window_hours: int = 24
    ) -> List[MetricData]:
        """
        Retrieve indexed historical metrics from S3.
        
        Args:
            metric_name: Metric name pattern to search
            time_window_hours: Hours to look back
            
        Returns:
            List of historical metric data points
        """
        try:
            # List objects in metrics prefix
            prefix = "metrics/"
            response = self.s3.list_objects_v2(
                Bucket=self.s3_bucket,
                Prefix=prefix,
                MaxKeys=100
            )
            
            metrics = []
            for obj in response.get('Contents', []):
                # Get object
                obj_response = self.s3.get_object(
                    Bucket=self.s3_bucket,
                    Key=obj['Key']
                )
                
                # Parse metrics
                data = json.loads(obj_response['Body'].read())
                for metric_dict in data.get('metrics', []):
                    if metric_name.lower() in metric_dict['metric_name'].lower():
                        # Reconstruct MetricData
                        from models.core_models import MetricType
                        metric = MetricData(
                            metric_name=metric_dict['metric_name'],
                            metric_type=MetricType(metric_dict['metric_type']),
                            value=metric_dict['value'],
                            unit=metric_dict['unit'],
                            timestamp=metric_dict['timestamp'],
                            dimensions=metric_dict['dimensions'],
                            source=metric_dict['source']
                        )
                        metrics.append(metric)
            
            logger.info(f"Retrieved {len(metrics)} historical metrics for {metric_name}")
            return metrics
            
        except ClientError as e:
            logger.error(f"Failed to index metrics: {e}")
            return []
    
    def create_embedding_document(
        self,
        content: str,
        doc_type: str,
        metadata: Dict[str, Any],
        prefix: str = "embeddings"
    ) -> str:
        """
        Create a document for embedding/indexing in Knowledge Base.
        
        Args:
            content: Document content
            doc_type: Document type
            metadata: Document metadata
            prefix: S3 prefix
            
        Returns:
            S3 object key
        """
        try:
            # Generate unique filename
            content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            filename = f"{doc_type}_{timestamp}_{content_hash}.txt"
            s3_key = f"{prefix}/{doc_type}/{filename}"
            
            # Create document with metadata header
            document_content = f"""---
type: {doc_type}
timestamp: {timestamp}
metadata: {json.dumps(metadata)}
---

{content}
"""
            
            # Upload to S3
            self.s3.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=document_content,
                ContentType='text/plain',
                Metadata={k: str(v) for k, v in metadata.items()}
            )
            
            logger.info(f"Created embedding document: s3://{self.s3_bucket}/{s3_key}")
            return s3_key
            
        except ClientError as e:
            logger.error(f"Failed to create embedding document: {e}")
            raise
