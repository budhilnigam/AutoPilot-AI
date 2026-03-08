"""
Local Knowledge Base Service

File-based knowledge base for local development and prototyping.
Easy to swap with AWS Bedrock Knowledge Base for production.
"""

import json
import logging
import os
import hashlib
from typing import Dict, List, Optional, Any
from datetime import datetime
import pickle
from pathlib import Path

from models.core_models import Configuration, MetricData

logger = logging.getLogger(__name__)


class LocalKnowledgeBase:
    """
    Local file-based Knowledge Base for development.
    
    Provides same interface as AWS-based KnowledgeBase for easy swapping.
    
    Storage:
    - Configurations: JSON files in local/configs/
    - Metrics: Pickle files in local/metrics/
    - Documents: Text files in local/documents/
    """
    
    def __init__(
        self,
        storage_path: str = None,
    ):
        """
        Initialize Local Knowledge Base.
        
        Args:
            storage_path: Path for local storage (default: ./local_kb)
        """
        self.storage_path = storage_path or os.getenv('LOCAL_KB_PATH', './local_kb')
        
        # Create storage directories
        self.configs_path = Path(self.storage_path) / 'configs'
        self.metrics_path = Path(self.storage_path) / 'metrics'
        self.documents_path = Path(self.storage_path) / 'documents'
        
        for path in [self.configs_path, self.metrics_path, self.documents_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Local Knowledge Base initialized at: {self.storage_path}")
    
    def store_configuration(
        self,
        config: Configuration,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store configuration locally.
        
        Args:
            config: Configuration object
            metadata: Additional metadata
            
        Returns:
            True if successful
        """
        try:
            # Create filename from config type and resource
            filename = f"{config.config_type}_{config.resource_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.configs_path / filename
            
            # Prepare data
            data = {
                'config_type': config.config_type,
                'resource_id': config.resource_id,
                'content': config.content,
                'source': config.source,
                'timestamp': datetime.utcnow().isoformat(),
                'metadata': metadata or {}
            }
            
            # Write to file
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Stored configuration: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store configuration: {e}")
            return False
    
    def store_metrics(
        self,
        metrics: List[MetricData],
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store metrics locally.
        
        Args:
            metrics: List of metric data points
            metadata: Additional metadata
            
        Returns:
            True if successful
        """
        try:
            if not metrics:
                return False
            
            # Create filename from first metric name and timestamp
            metric_name = metrics[0].metric_name
            filename = f"{metric_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pkl"
            filepath = self.metrics_path / filename
            
            # Prepare data
            data = {
                'metrics': [vars(m) for m in metrics],
                'timestamp': datetime.utcnow().isoformat(),
                'metadata': metadata or {}
            }
            
            # Write to pickle file
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Stored {len(metrics)} metrics: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store metrics: {e}")
            return False
    
    def query_context(
        self,
        query: str,
        max_results: int = 5,
        similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant context (simplified for local mode).
        
        Args:
            query: Search query
            max_results: Maximum number of results
            similarity_threshold: Minimum similarity score (ignored in simple mode)
            
        Returns:
            List of relevant documents with scores
        """
        try:
            results = []
            
            # Simple keyword matching for local mode
            query_lower = query.lower()
            keywords = set(query_lower.split())
            
            # Search through documents
            for filepath in self.documents_path.glob('*.txt'):
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                    
                    # Simple scoring based on keyword matches
                    content_lower = content.lower()
                    matches = sum(1 for keyword in keywords if keyword in content_lower)
                    score = matches / len(keywords) if keywords else 0
                    
                    if score > 0:
                        results.append({
                            'content': content,
                            'score': score,
                            'source': filepath.name,
                            'metadata': {'file': str(filepath)}
                        })
                
                except Exception as e:
                    logger.warning(f"Error reading {filepath}: {e}")
                    continue
            
            # Search through configurations
            for filepath in self.configs_path.glob('*.json'):
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    content_str = json.dumps(data.get('content', {}))
                    matches = sum(1 for keyword in keywords if keyword in content_str.lower())
                    score = matches / len(keywords) if keywords else 0
                    
                    if score > 0:
                        results.append({
                            'content': content_str,
                            'score': score,
                            'source': filepath.name,
                            'metadata': data.get('metadata', {})
                        })
                
                except Exception as e:
                    logger.warning(f"Error reading {filepath}: {e}")
                    continue
            
            # Sort by score and limit results
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:max_results]
            
        except Exception as e:
            logger.error(f"Failed to query context: {e}")
            return []
    
    def store_document(
        self,
        content: str,
        filename: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store a text document in the knowledge base.
        
        Args:
            content: Document content
            filename: Name for the document
            metadata: Additional metadata
            
        Returns:
            True if successful
        """
        try:
            filepath = self.documents_path / filename
            
            with open(filepath, 'w') as f:
                f.write(content)
            
            # Store metadata separately
            if metadata:
                meta_filepath = self.documents_path / f"{filename}.meta.json"
                with open(meta_filepath, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            logger.info(f"Stored document: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store document: {e}")
            return False
    
    def get_metrics_history(
        self,
        metric_name: str,
        hours_back: int = 24
    ) -> List[MetricData]:
        """
        Retrieve historical metrics.
        
        Args:
            metric_name: Name of the metric
            hours_back: How many hours of history to retrieve
            
        Returns:
            List of metric data points
        """
        try:
            all_metrics = []
            
            # Search through metric files
            pattern = f"{metric_name}_*.pkl"
            for filepath in self.metrics_path.glob(pattern):
                try:
                    with open(filepath, 'rb') as f:
                        data = pickle.load(f)
                    
                    # Convert back to MetricData objects
                    for metric_dict in data.get('metrics', []):
                        # Reconstruct MetricData (simplified)
                        all_metrics.append(metric_dict)
                        
                except Exception as e:
                    logger.warning(f"Error reading {filepath}: {e}")
                    continue
            
            # Sort by timestamp (most recent first)
            all_metrics.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            return all_metrics
            
        except Exception as e:
            logger.error(f"Failed to get metrics history: {e}")
            return []
    
    def health_check(self) -> bool:
        """
        Check if the local knowledge base is operational.
        
        Returns:
            True if healthy
        """
        try:
            # Check if directories exist and are writable
            test_file = self.storage_path / '.health_check'
            test_file.write_text('ok')
            test_file.unlink()
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
