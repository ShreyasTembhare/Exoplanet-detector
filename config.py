#!/usr/bin/env python3
"""
Configuration file for the Exoplanet Detection Pipeline

Centralizes all project settings, parameters, and constants.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
CACHE_DIR = PROJECT_ROOT / "cache"
EXPORTS_DIR = PROJECT_ROOT / "exports"
MODELS_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
CACHE_DIR.mkdir(exist_ok=True)
EXPORTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Database configuration
DATABASE_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', 'exoplanet_detector'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', ''),
    'pool_size': int(os.getenv('DB_POOL_SIZE', 10)),
    'max_overflow': int(os.getenv('DB_MAX_OVERFLOW', 20))
}

# Analysis parameters
ANALYSIS_CONFIG = {
    # Light curve processing
    'flatten_window_length': 101,
    'binning_factor': 1,
    'outlier_threshold': 3.0,
    
    # Period detection
    'min_period': 0.5,  # days
    'max_period': 50.0,  # days
    'period_resolution': 0.001,  # days
    
    # Transit detection
    'min_transit_duration': 0.5,  # hours
    'max_transit_duration': 24.0,  # hours
    'min_snr': 7.1,
    'min_transit_depth': 0.0001,  # 100 ppm
    
    # False positive analysis
    'odd_even_threshold': 0.1,
    'secondary_eclipse_threshold': 0.5,
    'stellar_variability_threshold': 0.1
}

# Machine learning configuration
ML_CONFIG = {
    'model_path': MODELS_DIR / "exoplanet_classifier.pkl",
    'feature_columns': [
        'bls_power', 'period', 'transit_depth', 'snr',
        'transit_duration', 'odd_even_ratio', 
        'secondary_eclipse', 'stellar_variability',
        'data_quality', 'observation_count'
    ],
    'confidence_threshold': 0.7,
    'cross_validation_folds': 5,
    'random_state': 42
}

# Star shortlist configuration
SHORTLIST_CONFIG = {
    'max_stars': 1000,
    'tmag_max': 12.0,
    'teff_min': 3000,  # K
    'teff_max': 6500,  # K
    'radius_max': 1.5,  # solar radii
    'cdpp_max': 100,  # ppm
    'cache_duration': 86400  # 24 hours in seconds
}

# Processing configuration
PROCESSING_CONFIG = {
    'max_retries': 3,
    'retry_delay': 2,  # seconds
    'timeout': 300,  # seconds
    'max_workers': 4,
    'chunk_size': 100,
    'memory_limit': 1024  # MB
}

# Cache configuration
CACHE_CONFIG = {
    'enabled': True,
    'max_size': 1000,  # MB
    'expiration': 86400,  # 24 hours in seconds
    'compression': True
}

# Export configuration
EXPORT_CONFIG = {
    'formats': ['txt', 'csv', 'json'],
    'include_plots': True,
    'plot_format': 'png',
    'plot_dpi': 300,
    'include_metadata': True
}

# UI configuration
UI_CONFIG = {
    'theme': 'light',
    'max_plot_points': 10000,
    'auto_refresh': True,
    'refresh_interval': 30,  # seconds
    'show_progress': True,
    'enable_animations': True
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': PROJECT_ROOT / "exoplanet_detector.log",
    'max_file_size': 10,  # MB
    'backup_count': 5
}

# External API configuration
API_CONFIG = {
    'nasa_exoplanet_archive': {
        'base_url': 'https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI',
        'timeout': 30,
        'retry_attempts': 3
    },
    'simbad': {
        'base_url': 'http://simbad.u-strasbg.fr/simbad/sim-id',
        'timeout': 15,
        'retry_attempts': 2
    },
    'tess_alerts': {
        'base_url': 'https://tess.mit.edu/observations/',
        'timeout': 20,
        'retry_attempts': 2
    }
}

# Quality control thresholds
QUALITY_THRESHOLDS = {
    'data_completeness': 0.8,  # 80% of data must be present
    'noise_level': 0.1,  # Maximum acceptable noise level
    'systematic_error': 0.05,  # Maximum systematic error
    'gap_fraction': 0.2,  # Maximum fraction of gaps
    'outlier_fraction': 0.1  # Maximum fraction of outliers
}

# Performance monitoring
PERFORMANCE_CONFIG = {
    'enabled': True,
    'metrics_retention': 30,  # days
    'alert_thresholds': {
        'processing_time': 300,  # seconds
        'memory_usage': 1024,  # MB
        'error_rate': 0.1  # 10%
    }
}

# Development configuration
DEV_CONFIG = {
    'debug_mode': os.getenv('DEBUG', 'False').lower() == 'true',
    'test_mode': os.getenv('TEST_MODE', 'False').lower() == 'true',
    'demo_data': True,
    'mock_external_apis': False
}

def get_config(section):
    """Get configuration for a specific section."""
    config_map = {
        'database': DATABASE_CONFIG,
        'analysis': ANALYSIS_CONFIG,
        'ml': ML_CONFIG,
        'shortlist': SHORTLIST_CONFIG,
        'processing': PROCESSING_CONFIG,
        'cache': CACHE_CONFIG,
        'export': EXPORT_CONFIG,
        'ui': UI_CONFIG,
        'logging': LOGGING_CONFIG,
        'api': API_CONFIG,
        'quality': QUALITY_THRESHOLDS,
        'performance': PERFORMANCE_CONFIG,
        'dev': DEV_CONFIG
    }
    return config_map.get(section, {})

def update_config(section, updates):
    """Update configuration for a specific section."""
    config_map = {
        'database': DATABASE_CONFIG,
        'analysis': ANALYSIS_CONFIG,
        'ml': ML_CONFIG,
        'shortlist': SHORTLIST_CONFIG,
        'processing': PROCESSING_CONFIG,
        'cache': CACHE_CONFIG,
        'export': EXPORT_CONFIG,
        'ui': UI_CONFIG,
        'logging': LOGGING_CONFIG,
        'api': API_CONFIG,
        'quality': QUALITY_THRESHOLDS,
        'performance': PERFORMANCE_CONFIG,
        'dev': DEV_CONFIG
    }
    
    if section in config_map:
        config_map[section].update(updates)
        return True
    return False

def validate_config():
    """Validate configuration parameters."""
    errors = []
    
    # Validate analysis parameters
    if ANALYSIS_CONFIG['min_period'] >= ANALYSIS_CONFIG['max_period']:
        errors.append("min_period must be less than max_period")
    
    if ANALYSIS_CONFIG['min_transit_duration'] >= ANALYSIS_CONFIG['max_transit_duration']:
        errors.append("min_transit_duration must be less than max_transit_duration")
    
    # Validate processing parameters
    if PROCESSING_CONFIG['max_workers'] <= 0:
        errors.append("max_workers must be positive")
    
    if PROCESSING_CONFIG['timeout'] <= 0:
        errors.append("timeout must be positive")
    
    # Validate quality thresholds (all should be between 0 and 1)
    for key, value in QUALITY_THRESHOLDS.items():
        if not 0 <= value <= 1:
            errors.append(f"{key} must be between 0 and 1")
    
    return errors

if __name__ == "__main__":
    # Test configuration
    print("Configuration validation:")
    errors = validate_config()
    if errors:
        print("Errors found:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration is valid")
    
    print(f"\nProject paths:")
    print(f"  Root: {PROJECT_ROOT}")
    print(f"  Cache: {CACHE_DIR}")
    print(f"  Exports: {EXPORTS_DIR}")
    print(f"  Models: {MODELS_DIR}") 