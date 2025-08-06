import time
import logging
import sqlite3
from datetime import datetime

class PerformanceMonitor:
    """Performance monitoring and metrics tracking."""
    
    def __init__(self, db_path='exoplanet_detector.db'):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.operation_times = {}
    
    def track_operation(self, operation_name):
        """Decorator to track operation performance."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    self._record_success(operation_name, duration)
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    self._record_error(operation_name, duration, str(e))
                    raise
            return wrapper
        return decorator
    
    def _record_success(self, operation, duration):
        """Record successful operation metrics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO performance_metrics (operation_name, duration, success, error_message)
                    VALUES (?, ?, ?, ?)
                ''', (operation, duration, True, None))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to record success metrics: {e}")
    
    def _record_error(self, operation, duration, error_message):
        """Record failed operation metrics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO performance_metrics (operation_name, duration, success, error_message)
                    VALUES (?, ?, ?, ?)
                ''', (operation, duration, False, error_message))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to record error metrics: {e}")
    
    def get_performance_report(self):
        """Get comprehensive performance report."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get overall statistics
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_operations,
                        AVG(duration) as avg_duration,
                        SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as error_count
                    FROM performance_metrics
                    WHERE timestamp >= datetime('now', '-24 hours')
                ''')
                
                stats = cursor.fetchone()
                if stats and stats[0] > 0:
                    total_ops, avg_duration, error_count = stats
                    error_rate = error_count / total_ops if total_ops > 0 else 0
                    
                    return {
                        'total_operations': total_ops,
                        'avg_processing_time': avg_duration or 0,
                        'error_rate': error_rate,
                        'success_rate': 1 - error_rate
                    }
                
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get performance report: {e}")
            return None 