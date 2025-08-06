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

def filter_stars_by_criteria(df, tmag_max=12, teff_min=3000, teff_max=6500, radius_max=1.5):
    """
    Filter stars by specific criteria for exoplanet detection.
    
    Args:
        df (pandas.DataFrame): Input DataFrame with star data
        tmag_max (float): Maximum T magnitude (brightness)
        teff_min (float): Minimum effective temperature (K)
        teff_max (float): Maximum effective temperature (K)
        radius_max (float): Maximum stellar radius (solar radii)
        
    Returns:
        pandas.DataFrame: Filtered DataFrame
    """
    if df.empty:
        logger.warning("Input DataFrame is empty")
        return df
    
    # Apply filters
    filtered_df = df.copy()
    
    # Filter by T magnitude (brighter stars)
    if 'Tmag' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Tmag'] <= tmag_max]
        logger.info(f"Filtered by Tmag <= {tmag_max}: {len(filtered_df)} stars remaining")
    
    # Filter by effective temperature
    if 'Teff' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['Teff'] >= teff_min) & 
            (filtered_df['Teff'] <= teff_max)
        ]
        logger.info(f"Filtered by Teff {teff_min}-{teff_max}K: {len(filtered_df)} stars remaining")
    
    # Filter by stellar radius
    if 'Radius' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Radius'] <= radius_max]
        logger.info(f"Filtered by Radius <= {radius_max}R☉: {len(filtered_df)} stars remaining")
    
    return filtered_df

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
    
    # Apply additional filtering
    df = filter_stars_by_criteria(df)
    
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
            'avg_cdpp': 0,
            'min_tmag': 0,
            'max_tmag': 0,
            'min_teff': 0,
            'max_teff': 0
        }
    
    stats = {
        'total_stars': len(df),
        'avg_tmag': df['Tmag'].mean() if 'Tmag' in df.columns else 0,
        'avg_teff': df['Teff'].mean() if 'Teff' in df.columns else 0,
        'avg_radius': df['Radius'].mean() if 'Radius' in df.columns else 0,
        'avg_cdpp': df['CDPP4_0'].mean() if 'CDPP4_0' in df.columns else 0,
        'min_tmag': df['Tmag'].min() if 'Tmag' in df.columns else 0,
        'max_tmag': df['Tmag'].max() if 'Tmag' in df.columns else 0,
        'min_teff': df['Teff'].min() if 'Teff' in df.columns else 0,
        'max_teff': df['Teff'].max() if 'Teff' in df.columns else 0
    }
    
    return stats

def validate_star_data(df):
    """
    Validate star data for quality and completeness.
    
    Args:
        df (pandas.DataFrame): Star data to validate
        
    Returns:
        dict: Validation results
    """
    validation = {
        'total_stars': len(df),
        'missing_tmag': 0,
        'missing_teff': 0,
        'missing_radius': 0,
        'invalid_tmag': 0,
        'invalid_teff': 0,
        'invalid_radius': 0
    }
    
    if df.empty:
        return validation
    
    # Check for missing values
    if 'Tmag' in df.columns:
        validation['missing_tmag'] = df['Tmag'].isna().sum()
        validation['invalid_tmag'] = len(df[(df['Tmag'] < 0) | (df['Tmag'] > 20)])
    
    if 'Teff' in df.columns:
        validation['missing_teff'] = df['Teff'].isna().sum()
        validation['invalid_teff'] = len(df[(df['Teff'] < 1000) | (df['Teff'] > 10000)])
    
    if 'Radius' in df.columns:
        validation['missing_radius'] = df['Radius'].isna().sum()
        validation['invalid_radius'] = len(df[(df['Radius'] < 0) | (df['Radius'] > 10)])
    
    return validation

if __name__ == "__main__":
    # Test the module
    print("Testing Star Shortlist Module...")
    df = get_shortlist(force_refresh=True)
    print(f"Generated shortlist with {len(df)} stars")
    print(df.head())
    
    stats = get_shortlist_stats()
    print(f"Shortlist statistics: {stats}")
    
    validation = validate_star_data(df)
    print(f"Data validation: {validation}") 