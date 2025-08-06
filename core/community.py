import logging
import sqlite3
from datetime import datetime

class CommunityManager:
    """Community features manager for sharing discoveries and annotations."""
    
    def __init__(self, db_path='exoplanet_detector.db'):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
    
    def add_annotation(self, target_name, annotation_type, content, user_id="anonymous"):
        """Add a community annotation for a target."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO community_annotations 
                    (target_name, annotation_type, content, user_id, rating)
                    VALUES (?, ?, ?, ?, ?)
                ''', (target_name, annotation_type, content, user_id, 0.0))
                conn.commit()
                
                self.logger.info(f"Added annotation for {target_name} by {user_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to add annotation: {e}")
            return False
    
    def get_annotations(self, target_name=None):
        """Get community annotations."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if target_name:
                    cursor.execute('''
                        SELECT * FROM community_annotations 
                        WHERE target_name = ? 
                        ORDER BY created_at DESC
                    ''', (target_name,))
                else:
                    cursor.execute('''
                        SELECT * FROM community_annotations 
                        ORDER BY created_at DESC
                    ''')
                
                columns = [description[0] for description in cursor.description]
                results = [dict(zip(columns, row)) for row in cursor.fetchall()]
                return results
                
        except Exception as e:
            self.logger.error(f"Failed to get annotations: {e}")
            return []
    
    def rate_annotation(self, annotation_id, rating):
        """Rate a community annotation."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE community_annotations 
                    SET rating = ? WHERE id = ?
                ''', (rating, annotation_id))
                conn.commit()
                
                self.logger.info(f"Rated annotation {annotation_id} with {rating}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to rate annotation: {e}")
            return False 