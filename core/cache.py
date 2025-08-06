import os
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timedelta

class CacheManager:
    """Cache manager for storing and retrieving analysis results."""
    
    def __init__(self, cache_dir='cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.cache_expiry_hours = 24
    
    def _get_cache_key(self, *args, **kwargs):
        """Generate a unique cache key from function arguments."""
        key_string = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_cached_result(self, cache_key):
        """Retrieve cached result if available and not expired."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                # Check if cache is expired
                file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                if file_age < timedelta(hours=self.cache_expiry_hours):
                    with open(cache_file, 'r') as f:
                        cached_data = json.load(f)
                        self.logger.info(f"Cache hit for key: {cache_key}")
                        return cached_data
                else:
                    # Remove expired cache
                    cache_file.unlink()
                    self.logger.info(f"Removed expired cache for key: {cache_key}")
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving cache: {e}")
            return None
    
    def cache_result(self, cache_key, result):
        """Cache analysis result."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            # Convert numpy arrays and other non-serializable objects
            serializable_result = self._make_serializable(result)
            
            with open(cache_file, 'w') as f:
                json.dump(serializable_result, f, default=str)
            
            self.logger.info(f"Cached result for key: {cache_key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error caching result: {e}")
            return False
    
    def _make_serializable(self, obj):
        """Convert object to JSON serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, 'value'):  # Astropy units
            return float(obj.value)
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj
    
    def clear_expired_cache(self):
        """Remove all expired cache files."""
        try:
            current_time = datetime.now()
            expired_count = 0
            
            for cache_file in self.cache_dir.glob("*.json"):
                file_age = current_time - datetime.fromtimestamp(cache_file.stat().st_mtime)
                if file_age > timedelta(hours=self.cache_expiry_hours):
                    cache_file.unlink()
                    expired_count += 1
            
            if expired_count > 0:
                self.logger.info(f"Cleared {expired_count} expired cache files")
            
            return expired_count
            
        except Exception as e:
            self.logger.error(f"Error clearing expired cache: {e}")
            return 0
    
    def clear_all_cache(self):
        """Clear all cache files."""
        try:
            cache_files = list(self.cache_dir.glob("*.json"))
            for cache_file in cache_files:
                cache_file.unlink()
            
            self.logger.info(f"Cleared all {len(cache_files)} cache files")
            return len(cache_files)
            
        except Exception as e:
            self.logger.error(f"Error clearing all cache: {e}")
            return 0 