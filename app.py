#!/usr/bin/env python3
"""
Advanced Exoplanet Detector - Main Application

A professional, modular exoplanet detection tool with advanced analysis capabilities.
"""

import streamlit as st
import logging
import warnings
warnings.filterwarnings('ignore')

# Import core modules
from core import (
    ExoplanetDatabase, ResilientProcessor, PerformanceMonitor, CacheManager,
    ParallelProcessor, MLModelManager, RealTimeMonitor, CommunityManager,
    EducationalManager, DataQualityAssessor, VisualizationManager, ExportManager
)

# Import analysis modules
from analysis import ExoplanetAnalyzer

# Import UI modules
from ui import StreamlitInterface

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('exoplanet_detector.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def initialize_components():
    """Initialize all application components."""
    logger = setup_logging()
    
    # Initialize core components
    db = ExoplanetDatabase()
    resilient_processor = ResilientProcessor()
    performance_monitor = PerformanceMonitor()
    cache_manager = CacheManager()
    parallel_processor = ParallelProcessor()
    ml_manager = MLModelManager()
    real_time_monitor = RealTimeMonitor()
    community_manager = CommunityManager()
    educational_manager = EducationalManager()
    quality_assessor = DataQualityAssessor()
    viz_manager = VisualizationManager()
    export_manager = ExportManager()
    
    # Initialize analyzer
    analyzer = ExoplanetAnalyzer(
        resilient_processor=resilient_processor,
        quality_assessor=quality_assessor,
        cache_manager=cache_manager,
        db=db,
        ml_manager=ml_manager
    )
    
    # Initialize interface
    interface = StreamlitInterface(
        analyzer=analyzer,
        db=db,
        export_manager=export_manager,
        viz_manager=viz_manager,
        community_manager=community_manager,
        educational_manager=educational_manager
    )
    
    logger.info("All components initialized successfully")
    
    return interface

def main():
    """Main application entry point."""
    try:
        # Initialize components
        interface = initialize_components()
        
        # Run the interface
        interface.run()
        
    except Exception as e:
        st.error(f"Application error: {e}")
        logging.error(f"Application error: {e}")

if __name__ == "__main__":
    main()