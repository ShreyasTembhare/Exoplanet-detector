#!/usr/bin/env python3
"""
Star Shortlist Module

Queries TESS Input Catalog for stars likely to host exoplanets based on specific criteria.
"""

import pandas as pd
import numpy as np
import logging
from astroquery.mast import Catalogs
from astroquery.exceptions import ResolverError
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def query_tess_catalog(max_stars=1000):
    """
    Query TESS Input Catalog for stars matching exoplanet host criteria.
    
    Criteria:
    - Tmag < 12 (bright enough for good photometry)
    - Radius < 1.5 R☉ (smaller stars = deeper transits)
    - 3000 K < Teff < 6500 K (main sequence stars)
    - CDPP4_0 < 100 ppm (low noise)
    
    Returns:
        pandas.DataFrame: DataFrame with matching stars
    """
    try:
        logger.info("Querying TESS Input Catalog...")
        
        # Query TESS Input Catalog with our criteria
        catalog_data = Catalogs.query_criteria(
            catalog="Tic",
            Tmag=[0, 12],
            radius=[0, 1.5],
            Teff=[3000, 6500]
        )
        
        # Limit to max_stars to avoid overwhelming results
        if len(catalog_data) > max_stars:
            catalog_data = catalog_data[:max_stars]
        
        logger.info(f"Found {len(catalog_data)} stars matching criteria")
        
        # Convert to pandas DataFrame
        df = catalog_data.to_pandas()
        
        # Select relevant columns
        columns_to_keep = ['ID', 'Tmag', 'Teff', 'radius']
        available_columns = [col for col in columns_to_keep if col in df.columns]
        
        df = df[available_columns]
        
        # Rename columns for consistency
        column_mapping = {
            'ID': 'ID',
            'Tmag': 'Tmag',
            'Teff': 'Teff',
            'radius': 'Radius'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Add any missing columns with NaN values
        for col in columns_to_keep:
            if col not in df.columns:
                df[col] = np.nan
        
        # Add CDPP4_0 column with NaN values (not available in catalog query)
        df['CDPP4_0'] = np.nan
        
        return df
        
    except Exception as e:
        logger.error(f"Error querying TESS catalog: {e}")
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=['ID', 'Tmag', 'Teff', 'Radius', 'CDPP4_0'])

def get_shortlist(force_refresh=False):
    """
    Get the star shortlist DataFrame.
    
    Args:
        force_refresh (bool): If True, regenerate the shortlist even if file exists
        
    Returns:
        pandas.DataFrame: DataFrame with star shortlist
    """
    csv_file = 'star_shortlist.csv'
    
    # Check if file exists and we're not forcing refresh
    if os.path.exists(csv_file) and not force_refresh:
        try:
            logger.info(f"Loading existing shortlist from {csv_file}")
            df = pd.read_csv(csv_file)
            logger.info(f"Loaded {len(df)} stars from existing shortlist")
            return df
        except Exception as e:
            logger.warning(f"Error loading existing shortlist: {e}")
            # Continue to regenerate
    
    # Generate new shortlist
    logger.info("Generating new star shortlist...")
    df = query_tess_catalog()
    
    if not df.empty:
        # Save to CSV
        try:
            df.to_csv(csv_file, index=False)
            logger.info(f"Saved shortlist to {csv_file}")
        except Exception as e:
            logger.error(f"Error saving shortlist: {e}")
    else:
        logger.warning("No stars found matching criteria")
    
    return df

def get_shortlist_stats():
    """
    Get statistics about the current shortlist.
    
    Returns:
        dict: Statistics about the shortlist
    """
    df = get_shortlist()
    
    if df.empty:
        return {
            'total_stars': 0,
            'avg_tmag': 0,
            'avg_teff': 0,
            'avg_radius': 0,
            'avg_cdpp': 0
        }
    
    stats = {
        'total_stars': len(df),
        'avg_tmag': df['Tmag'].mean() if 'Tmag' in df.columns else 0,
        'avg_teff': df['Teff'].mean() if 'Teff' in df.columns else 0,
        'avg_radius': df['Radius'].mean() if 'Radius' in df.columns else 0,
        'avg_cdpp': df['CDPP4_0'].mean() if 'CDPP4_0' in df.columns else 0
    }
    
    return stats

if __name__ == "__main__":
    # Test the module
    print("Testing Star Shortlist Module...")
    df = get_shortlist(force_refresh=True)
    print(f"Generated shortlist with {len(df)} stars")
    print(df.head())
    
    stats = get_shortlist_stats()
    print(f"Shortlist statistics: {stats}") 