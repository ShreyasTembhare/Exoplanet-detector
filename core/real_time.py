import threading
import time
import logging
import sqlite3
from datetime import datetime

class RealTimeMonitor:
    """Real-time monitoring for exoplanet detection targets."""
    
    def __init__(self, db_path='exoplanet_detector.db'):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.monitoring_threads = {}
        self.stop_monitoring_flag = {}
    
    def start_monitoring(self, target_name, alert_thresholds=None):
        """Start monitoring a target for real-time alerts."""
        try:
            if target_name in self.monitoring_threads and self.monitoring_threads[target_name].is_alive():
                self.logger.warning(f"Already monitoring {target_name}")
                return False
            
            # Set default thresholds if none provided
            if alert_thresholds is None:
                alert_thresholds = {
                    'snr_threshold': 7.0,
                    'transit_depth_threshold': 0.001,
                    'period_change_threshold': 0.1
                }
            
            # Save to database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO monitoring_targets 
                    (target_name, alert_thresholds, is_active, last_check)
                    VALUES (?, ?, ?, ?)
                ''', (target_name, str(alert_thresholds), True, datetime.now()))
                conn.commit()
            
            # Start monitoring thread
            self.stop_monitoring_flag[target_name] = False
            thread = threading.Thread(
                target=self._monitor_target,
                args=(target_name, alert_thresholds),
                daemon=True
            )
            thread.start()
            self.monitoring_threads[target_name] = thread
            
            self.logger.info(f"Started monitoring {target_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring {target_name}: {e}")
            return False
    
    def _monitor_target(self, target_name, alert_thresholds):
        """Monitor target in background thread."""
        while not self.stop_monitoring_flag.get(target_name, False):
            try:
                # Perform quick analysis
                results = self._quick_analysis(target_name)
                
                if results:
                    # Check for alerts
                    alerts = self._check_alerts(target_name, results, alert_thresholds)
                    
                    if alerts:
                        self._handle_alerts(target_name, alerts)
                
                # Update last check time
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        UPDATE monitoring_targets SET last_check = ? WHERE target_name = ?
                    ''', (datetime.now(), target_name))
                    conn.commit()
                
                # Wait before next check
                time.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error monitoring {target_name}: {e}")
                time.sleep(60)  # Wait 1 minute before retry
    
    def _quick_analysis(self, target_name):
        """Perform quick analysis for monitoring."""
        try:
            # This would typically involve a lightweight analysis
            # For now, return a simple structure
            return {
                'target_name': target_name,
                'snr': 8.5,  # Simulated values
                'transit_depth': 0.002,
                'period': 3.5,
                'timestamp': datetime.now()
            }
        except Exception as e:
            self.logger.error(f"Quick analysis failed for {target_name}: {e}")
            return None
    
    def _check_alerts(self, target_name, results, thresholds):
        """Check if results trigger any alerts."""
        alerts = []
        
        try:
            snr = results.get('snr', 0)
            transit_depth = results.get('transit_depth', 0)
            
            if snr > thresholds.get('snr_threshold', 7.0):
                alerts.append({
                    'type': 'high_snr',
                    'message': f"High SNR detected: {snr:.2f}",
                    'severity': 'high'
                })
            
            if transit_depth > thresholds.get('transit_depth_threshold', 0.001):
                alerts.append({
                    'type': 'deep_transit',
                    'message': f"Deep transit detected: {transit_depth:.4f}",
                    'severity': 'medium'
                })
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error checking alerts for {target_name}: {e}")
            return []
    
    def _handle_alerts(self, target_name, alerts):
        """Handle triggered alerts."""
        try:
            for alert in alerts:
                self.logger.warning(f"ALERT for {target_name}: {alert['message']}")
                
                # Here you could send notifications, emails, etc.
                # For now, just log the alert
                
        except Exception as e:
            self.logger.error(f"Error handling alerts for {target_name}: {e}")
    
    def stop_monitoring(self, target_name):
        """Stop monitoring a target."""
        try:
            if target_name in self.monitoring_threads:
                self.stop_monitoring_flag[target_name] = True
                self.monitoring_threads[target_name].join(timeout=5)
                
                # Update database
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        UPDATE monitoring_targets SET is_active = FALSE WHERE target_name = ?
                    ''', (target_name,))
                    conn.commit()
                
                self.logger.info(f"Stopped monitoring {target_name}")
                return True
            else:
                self.logger.warning(f"Not monitoring {target_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error stopping monitoring for {target_name}: {e}")
            return False
    
    def get_monitoring_status(self):
        """Get status of all monitored targets."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT target_name, is_active, last_check, alert_thresholds
                    FROM monitoring_targets
                    ORDER BY last_check DESC
                ''')
                
                columns = [description[0] for description in cursor.description]
                results = [dict(zip(columns, row)) for row in cursor.fetchall()]
                return results
                
        except Exception as e:
            self.logger.error(f"Failed to get monitoring status: {e}")
            return [] 