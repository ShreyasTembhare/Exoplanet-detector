#!/usr/bin/env python3
"""
Analysis & Persistence Module

Handles SQLite database operations, light curve analysis, and BLS transit detection
for the star shortlist pipeline.
"""

import sqlite3
import time
import logging
import pandas as pd
import numpy as np
import lightkurve as lk
from lightkurve import LightkurveWarning
import warnings
from astropy import units as u
import os

# Suppress Lightkurve warnings
warnings.filterwarnings('ignore', category=LightkurveWarning)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalysisPipeline:
    """Handles analysis and persistence for star shortlist pipeline."""
    
    def __init__(self, db_path='candidates.db'):
        """Initialize the analysis pipeline."""
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize the SQLite database with candidates table."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS candidates (
                    star_id TEXT PRIMARY KEY,
                    period REAL,
                    depth REAL,
                    duration REAL,
                    detected_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    bls_power REAL,
                    snr REAL,
                    mission TEXT,
                    status TEXT DEFAULT 'pending'
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info(f"Database initialized: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def is_seen(self, star_id):
        """Check if a star has already been analyzed."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT star_id FROM candidates WHERE star_id = ?', (star_id,))
            result = cursor.fetchone()
            
            conn.close()
            return result is not None
            
        except Exception as e:
            logger.error(f"Error checking if star {star_id} is seen: {e}")
            return False
    
    def record_candidate(self, star_id, period, depth, duration, bls_power=0, snr=0, mission='TESS', status='candidate'):
        """Record a candidate in the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO candidates 
                (star_id, period, depth, duration, bls_power, snr, mission, status, detected_on)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ''', (star_id, period, depth, duration, bls_power, snr, mission, status))
            
            conn.commit()
            conn.close()
            logger.info(f"Recorded candidate: {star_id}")
            
        except Exception as e:
            logger.error(f"Error recording candidate {star_id}: {e}")
    
    def analyze_star(self, star_id, mission='TESS'):
        """
        Analyze a single star for transit signals.
        
        Args:
            star_id (str): Star ID (TIC, KIC, etc.)
            mission (str): Mission name (TESS, Kepler, etc.)
            
        Returns:
            tuple or None: (period, depth, duration) if transit found, None otherwise
        """
        try:
            logger.info(f"Analyzing star {star_id} from {mission}")
            
            # Search for light curve data
            if mission.upper() == 'TESS':
                search_result = lk.search_lightcurve(f"TIC {star_id}", mission='TESS')
            elif mission.upper() == 'KEPLER':
                search_result = lk.search_lightcurve(f"KIC {star_id}", mission='Kepler')
            else:
                search_result = lk.search_lightcurve(star_id)
            
            if len(search_result) == 0:
                logger.warning(f"No light curve data found for {star_id}")
                return None
            
            # Download the first available light curve
            lc = search_result[0].download()
            
            # Flatten the light curve
            lc_flat = lc.flatten()
            
            # Run BLS search
            bls = lc_flat.to_periodogram(method='bls', 
                                        minimum_period=0.5*u.day, 
                                        maximum_period=20*u.day,
                                        frequency_factor=0.01)
            
            # Get the best period
            best_period = bls.period_at_max_power
            
            # Check if the signal is significant (power >= 7.0)
            max_power = bls.power_at_max_power
            if max_power < 7.0:
                logger.info(f"Signal not significant for {star_id} (power: {max_power:.2f})")
                return None
            
            # Fold the light curve at the best period
            lc_folded = lc_flat.fold(period=best_period)
            
            # Calculate transit depth and duration
            flux_values = lc_folded.flux.value if hasattr(lc_folded.flux, 'value') else lc_folded.flux
            transit_depth = 1.0 - np.min(flux_values)
            
            # Estimate duration (simplified)
            transit_duration = 0.1  # Default duration in days
            
            # Calculate SNR
            clean_flux = lc_flat.flux.value if hasattr(lc_flat.flux, 'value') else lc_flat.flux
            snr = transit_depth / np.std(clean_flux)
            
            # Extract numerical values
            period_value = best_period.value if hasattr(best_period, 'value') else float(best_period)
            
            logger.info(f"Found transit signal for {star_id}: period={period_value:.2f}d, depth={transit_depth:.4f}")
            
            return {
                'period': period_value,
                'depth': transit_depth,
                'duration': transit_duration,
                'bls_power': max_power,
                'snr': snr
            }
            
        except Exception as e:
            logger.error(f"Error analyzing star {star_id}: {e}")
            return None
    
    def process_shortlist(self, shortlist_df, max_stars=None, progress_callback=None):
        """
        Process the star shortlist and analyze each star.
        
        Args:
            shortlist_df (pandas.DataFrame): DataFrame with star IDs
            max_stars (int): Maximum number of stars to process
            progress_callback (callable): Optional callback for progress updates
            
        Returns:
            list: List of newly discovered candidates
        """
        candidates = []
        total_stars = len(shortlist_df)
        
        if max_stars:
            total_stars = min(total_stars, max_stars)
        
        logger.info(f"Starting analysis of {total_stars} stars")
        
        for i, row in shortlist_df.iterrows():
            if max_stars and i >= max_stars:
                break
                
            star_id = str(row['ID'])
            
            # Skip if already analyzed
            if self.is_seen(star_id):
                logger.info(f"Skipping {star_id} (already analyzed)")
                continue
            
            # Analyze the star
            result = self.analyze_star(star_id)
            
            if result:
                # Record the candidate
                self.record_candidate(
                    star_id=star_id,
                    period=result['period'],
                    depth=result['depth'],
                    duration=result['duration'],
                    bls_power=result['bls_power'],
                    snr=result['snr']
                )
                candidates.append({
                    'star_id': star_id,
                    **result
                })
                logger.info(f"New candidate found: {star_id}")
            
            # Progress callback
            if progress_callback:
                progress_callback(i + 1, total_stars, star_id, result is not None)
            
            # Small delay to avoid overwhelming the server
            time.sleep(0.1)
        
        logger.info(f"Analysis complete. Found {len(candidates)} new candidates")
        return candidates
    
    def get_all_candidates(self):
        """Get all candidates from the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query('SELECT * FROM candidates ORDER BY detected_on DESC', conn)
            conn.close()
            return df
        except Exception as e:
            logger.error(f"Error getting candidates: {e}")
            return pd.DataFrame()
    
    def get_candidate_stats(self):
        """Get statistics about candidates."""
        df = self.get_all_candidates()
        
        if df.empty:
            return {
                'total_candidates': 0,
                'avg_period': 0,
                'avg_depth': 0,
                'avg_snr': 0,
                'mission_distribution': {}
            }
        
        stats = {
            'total_candidates': len(df),
            'avg_period': df['period'].mean() if 'period' in df.columns else 0,
            'avg_depth': df['depth'].mean() if 'depth' in df.columns else 0,
            'avg_snr': df['snr'].mean() if 'snr' in df.columns else 0,
            'mission_distribution': df['mission'].value_counts().to_dict() if 'mission' in df.columns else {}
        }
        
        return stats
    
    def clear_database(self):
        """Clear all candidates from the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM candidates')
            conn.commit()
            conn.close()
            logger.info("Database cleared")
        except Exception as e:
            logger.error(f"Error clearing database: {e}")

def process_shortlist(shortlist_df, max_stars=None, progress_callback=None):
    """
    Convenience function to process the shortlist.
    
    Args:
        shortlist_df (pandas.DataFrame): DataFrame with star IDs
        max_stars (int): Maximum number of stars to process
        progress_callback (callable): Optional callback for progress updates
        
    Returns:
        list: List of newly discovered candidates
    """
    pipeline = AnalysisPipeline()
    return pipeline.process_shortlist(shortlist_df, max_stars, progress_callback)

if __name__ == "__main__":
    # Test the module
    print("Testing Analysis Pipeline...")
    
    # Create test shortlist
    test_df = pd.DataFrame({
        'ID': ['123456789', '987654321'],
        'Tmag': [10.5, 11.2],
        'Teff': [5500, 4800],
        'Radius': [1.0, 0.8],
        'CDPP4_0': [50, 75]
    })
    
    pipeline = AnalysisPipeline()
    candidates = pipeline.process_shortlist(test_df, max_stars=2)
    print(f"Found {len(candidates)} candidates")
    
    stats = pipeline.get_candidate_stats()
    print(f"Candidate statistics: {stats}") 