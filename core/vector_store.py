from typing import Dict, Any

def get_status(self) -> Dict[str, Any]:
    """Get vector store status and statistics."""
    try:
        status = {
            'is_connected': True,
            'is_compatible': True,
            'db_type': self.db_type,
            'version': self.version,
            'in_memory': self.in_memory,
            'docs_count': len(self.documents) if hasattr(self, 'documents') else 0,
            'collection_name': self.collection_name if hasattr(self, 'collection_name') else None,
            'persist_directory': self.persist_directory if hasattr(self, 'persist_directory') else None,
            'embedding_dim': self.embedding_dimension if hasattr(self, 'embedding_dimension') else None,
            'index_size': self.get_index_size(),
            'query_latency': self.get_average_query_latency(),
            'cache_size': self.get_cache_size(),
            'hit_rate': self.get_cache_hit_rate()
        }
        return status
    except Exception as e:
        logger.error(f"Error getting vector store status: {e}")
        return {
            'is_connected': False,
            'is_compatible': False,
            'error': str(e)
        }

def get_index_size(self) -> int:
    """Get the size of the vector index in bytes."""
    try:
        if hasattr(self, 'index'):
            return self.index.get_size()
        return 0
    except Exception as e:
        logger.error(f"Error getting index size: {e}")
        return 0

def get_average_query_latency(self) -> float:
    """Get the average query latency in seconds."""
    try:
        if hasattr(self, 'query_latencies'):
            return sum(self.query_latencies) / len(self.query_latencies)
        return 0.0
    except Exception as e:
        logger.error(f"Error getting average query latency: {e}")
        return 0.0

def get_cache_size(self) -> int:
    """Get the size of the cache in bytes."""
    try:
        if hasattr(self, 'cache'):
            return self.cache.get_size()
        return 0
    except Exception as e:
        logger.error(f"Error getting cache size: {e}")
        return 0

def get_cache_hit_rate(self) -> float:
    """Get the cache hit rate as a percentage."""
    try:
        if hasattr(self, 'cache_hits') and hasattr(self, 'cache_misses'):
            total = self.cache_hits + self.cache_misses
            return (self.cache_hits / total * 100) if total > 0 else 0.0
        return 0.0
    except Exception as e:
        logger.error(f"Error getting cache hit rate: {e}")
        return 0.0 