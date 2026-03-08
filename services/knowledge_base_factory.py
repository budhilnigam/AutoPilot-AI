"""
Knowledge Base Factory

Creates appropriate Knowledge Base instance based on configuration.
Enables easy switching between local (development) and AWS (production) modes.
"""

import logging
from typing import Optional

from config import config
from services.local_knowledge_base import LocalKnowledgeBase
from services.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


def create_knowledge_base(
    use_local: Optional[bool] = None,
    **kwargs
) -> object:
    """
    Factory function to create appropriate Knowledge Base instance.
    
    Args:
        use_local: Override config.USE_LOCAL_KB if provided
        **kwargs: Additional arguments to pass to the KB constructor
        
    Returns:
        Knowledge Base instance (Local or AWS)
    """
    use_local_kb = use_local if use_local is not None else config.USE_LOCAL_KB
    
    if use_local_kb:
        logger.info("Creating Local Knowledge Base for development")
        storage_path = kwargs.get('storage_path', config.LOCAL_KB_PATH)
        return LocalKnowledgeBase(storage_path=storage_path)
    else:
        logger.info("Creating AWS Bedrock Knowledge Base for production")
        return KnowledgeBase(
            region_name=kwargs.get('region_name', config.AWS_REGION),
            s3_bucket=kwargs.get('s3_bucket', config.S3_BUCKET_NAME),
            knowledge_base_id=kwargs.get('knowledge_base_id', config.KNOWLEDGE_BASE_ID),
        )


class KnowledgeBaseInterface:
    """
    Unified interface for Knowledge Base operations.
    
    This ensures both Local and AWS implementations have the same interface.
    """
    
    def __init__(self, use_local: Optional[bool] = None):
        """
        Initialize KB interface with appropriate implementation.
        
        Args:
            use_local: Use local KB instead of AWS
        """
        self._kb = create_knowledge_base(use_local)
        self.is_local = isinstance(self._kb, LocalKnowledgeBase)
    
    def store_configuration(self, config, **kwargs):
        """Store configuration in KB"""
        return self._kb.store_configuration(config, **kwargs)
    
    def store_metrics(self, metrics, **kwargs):
        """Store metrics in KB"""
        return self._kb.store_metrics(metrics, **kwargs)
    
    def query_context(self, query, max_results=5, **kwargs):
        """Query KB for relevant context"""
        if self.is_local:
            return self._kb.query_context(
                query,
                max_results=max_results,
                similarity_threshold=kwargs.get('min_score', 0.0)
            )
        else:
            return self._kb.query_context(
                query,
                max_results=max_results,
                min_score=kwargs.get('min_score', 0.6)
            )
    
    def health_check(self):
        """Check if KB is operational"""
        if hasattr(self._kb, 'health_check'):
            return self._kb.health_check()
        # For AWS KB, try a simple operation
        try:
            if self.is_local:
                return self._kb.health_check()
            else:
                # For AWS, check if we can access S3 bucket
                import boto3
                s3 = boto3.client('s3')
                s3.head_bucket(Bucket=self._kb.s3_bucket)
                return True
        except Exception as e:
            logger.error(f"Knowledge Base health check failed: {e}")
            return False
    
    def store_document(self, content, filename, **kwargs):
        """Store a document in KB"""
        if hasattr(self._kb, 'store_document'):
            return self._kb.store_document(content, filename, **kwargs)
        else:
            logger.warning("store_document not implemented for this KB type")
            return False
    
    def __getattr__(self, name):
        """Delegate unknown methods to underlying KB implementation"""
        return getattr(self._kb, name)
