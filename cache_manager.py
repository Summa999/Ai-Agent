import redis
import json
import pickle
import pandas as pd
from typing import Any, Optional, Union
from datetime import datetime, timedelta
import logging

class CacheManager:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize Redis connection
        self.redis_client = redis.Redis(
            host=config.get('redis_host', 'localhost'),
            port=config.get('redis_port', 6379),
            db=config.get('redis_db', 0),
            decode_responses=False
        )
        
        # Test connection
        try:
            self.redis_client.ping()
            self.logger.info("Connected to Redis cache")
        except:
            self.logger.warning("Redis not available, using in-memory cache")
            self.redis_client = None
            self.memory_cache = {}
    
    def set(self, key: str, value: Any, expire_seconds: int = 3600):
        """Set value in cache with expiration"""
        try:
            if self.redis_client:
                # Serialize value
                if isinstance(value, pd.DataFrame):
                    serialized = pickle.dumps(value)
                else:
                    serialized = json.dumps(value).encode('utf-8')
                
                self.redis_client.setex(key, expire_seconds, serialized)
            else:
                # Use in-memory cache
                self.memory_cache[key] = {
                    'value': value,
                    'expires': datetime.now() + timedelta(seconds=expire_seconds)
                }
                
        except Exception as e:
            self.logger.error(f"Cache set error: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            if self.redis_client:
                value = self.redis_client.get(key)
                
                if value:
                    # Try to deserialize as DataFrame first
                    try:
                        return pickle.loads(value)
                    except:
                        # Try JSON
                        return json.loads(value.decode('utf-8'))
            else:
                # Use in-memory cache
                if key in self.memory_cache:
                    cached = self.memory_cache[key]
                    if cached['expires'] > datetime.now():
                        return cached['value']
                    else:
                        del self.memory_cache[key]
                        
        except Exception as e:
            self.logger.error(f"Cache get error: {e}")
            
        return None
    
    def delete(self, key: str):
        """Delete key from cache"""
        try:
            if self.redis_client:
                self.redis_client.delete(key)
            else:
                if key in self.memory_cache:
                    del self.memory_cache[key]
                    
        except Exception as e:
            self.logger.error(f"Cache delete error: {e}")
    
    def clear_pattern(self, pattern: str):
        """Clear all keys matching pattern"""
        try:
            if self.redis_client:
                for key in self.redis_client.scan_iter(match=pattern):
                    self.redis_client.delete(key)
            else:
                # In-memory cache
                keys_to_delete = [k for k in self.memory_cache.keys() 
                                 if pattern.replace('*', '') in k]
                for key in keys_to_delete:
                    del self.memory_cache[key]
                    
        except Exception as e:
            self.logger.error(f"Cache clear pattern error: {e}")
    
    def get_or_set(self, key: str, func, expire_seconds: int = 3600) -> Any:
        """Get from cache or compute and set"""
        value = self.get(key)
        
        if value is None:
            value = func()
            self.set(key, value, expire_seconds)
        
        return value
    
    def cleanup_expired(self):
        """Clean up expired entries from in-memory cache"""
        if not self.redis_client and hasattr(self, 'memory_cache'):
            current_time = datetime.now()
            expired_keys = [
                key for key, data in self.memory_cache.items()
                if data['expires'] < current_time
            ]
            
            for key in expired_keys:
                del self.memory_cache[key]
            
            if expired_keys:
                self.logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")