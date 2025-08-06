#!/usr/bin/env python3
"""
Test script for Star Shortlist & Analysis functionality.
"""

import pandas as pd
from shortlist import get_shortlist, get_shortlist_stats
from analysis_pipeline import AnalysisPipeline

def test_shortlist():
    """Test the shortlist functionality."""
    print("Testing Star Shortlist Module...")
    
    # Get shortlist
    df = get_shortlist(force_refresh=True)
    print(f"Generated shortlist with {len(df)} stars")
    
    if not df.empty:
        print("\nFirst 5 stars:")
        print(df.head())
    
    # Get statistics
    stats = get_shortlist_stats()
    print(f"\nShortlist statistics: {stats}")

def test_analysis_pipeline():
    """Test the analysis pipeline."""
    print("\nTesting Analysis Pipeline...")
    
    # Initialize pipeline
    pipeline = AnalysisPipeline()
    
    # Create test shortlist
    test_df = pd.DataFrame({
        'ID': ['123456789', '987654321'],
        'Tmag': [10.5, 11.2],
        'Teff': [5500, 4800],
        'Radius': [1.0, 0.8],
        'CDPP4_0': [50, 75]
    })
    
    print(f"Testing with {len(test_df)} stars")
    
    # Process shortlist (this will likely fail for synthetic IDs, but tests the structure)
    try:
        candidates = pipeline.process_shortlist(test_df, max_stars=2)
        print(f"Found {len(candidates)} candidates")
    except Exception as e:
        print(f"Analysis failed (expected for synthetic IDs): {e}")
    
    # Test database operations
    stats = pipeline.get_candidate_stats()
    print(f"Candidate statistics: {stats}")

if __name__ == "__main__":
    print("=== Star Shortlist & Analysis Test ===\n")
    
    test_shortlist()
    test_analysis_pipeline()
    
    print("\n=== Test Complete ===")
    print("To use the full functionality, run: streamlit run app.py")
    print("Then navigate to 'Star Shortlist & Analysis' in the sidebar.") 