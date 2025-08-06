import logging
import numpy as np
from astropy import units as u

class DataQualityAssessor:
    """Data quality assessment for light curves."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def assess_light_curve_quality(self, lc):
        """Assess the quality of a light curve."""
        try:
            quality_scores = {}
            
            # Completeness
            quality_scores['completeness'] = self._assess_completeness(lc)
            
            # Noise level
            quality_scores['noise_level'] = self._assess_noise_level(lc)
            
            # Gap analysis
            quality_scores['max_gap'] = self._assess_gaps(lc)
            
            # Systematic errors
            quality_scores['systematic_errors'] = self._detect_systematic_errors(lc)
            
            # Overall score
            quality_scores['overall_score'] = self._calculate_overall_score(quality_scores)
            
            return quality_scores
            
        except Exception as e:
            self.logger.error(f"Failed to assess light curve quality: {e}")
            return {
                'completeness': 0.0,
                'noise_level': 0.0,
                'max_gap': 0.0,
                'systematic_errors': 0.0,
                'overall_score': 0.0
            }
    
    def _assess_completeness(self, lc):
        """Assess data completeness."""
        try:
            if hasattr(lc.time, 'value'):
                time_values = lc.time.value
            else:
                time_values = lc.time
            
            if len(time_values) < 100:
                return 0.3  # Low completeness
            elif len(time_values) < 1000:
                return 0.7  # Medium completeness
            else:
                return 1.0  # High completeness
                
        except Exception as e:
            self.logger.error(f"Error assessing completeness: {e}")
            return 0.5
    
    def _assess_noise_level(self, lc):
        """Assess noise level in the light curve."""
        try:
            if hasattr(lc.flux, 'value'):
                flux_values = lc.flux.value
            else:
                flux_values = lc.flux
            
            # Calculate noise as standard deviation
            noise = np.std(flux_values)
            
            # Normalize noise level (lower is better)
            if noise < 0.001:
                return 1.0  # Very low noise
            elif noise < 0.01:
                return 0.8  # Low noise
            elif noise < 0.1:
                return 0.5  # Medium noise
            else:
                return 0.2  # High noise
                
        except Exception as e:
            self.logger.error(f"Error assessing noise level: {e}")
            return 0.5
    
    def _assess_gaps(self, lc):
        """Assess gaps in the light curve."""
        try:
            if hasattr(lc.time, 'value'):
                time_values = lc.time.value
            else:
                time_values = lc.time
            
            if len(time_values) < 2:
                return 0.0
            
            # Calculate time differences
            time_diffs = np.diff(time_values)
            max_gap = np.max(time_diffs)
            
            # Normalize gap size (smaller is better)
            if max_gap < 1.0:
                return 1.0  # Small gaps
            elif max_gap < 10.0:
                return 0.7  # Medium gaps
            else:
                return 0.3  # Large gaps
                
        except Exception as e:
            self.logger.error(f"Error assessing gaps: {e}")
            return 0.5
    
    def _detect_systematic_errors(self, lc):
        """Detect systematic errors in the light curve."""
        try:
            if hasattr(lc.flux, 'value'):
                flux_values = lc.flux.value
            else:
                flux_values = lc.flux
            
            # Check for NaN or infinite values
            if not np.all(np.isfinite(flux_values)):
                return 0.0  # Poor quality due to invalid data
            
            # Check for constant values (no variation)
            if np.std(flux_values) < 1e-6:
                return 0.2  # Poor quality due to no variation
            
            # Check for extreme outliers
            mean_flux = np.mean(flux_values)
            std_flux = np.std(flux_values)
            outliers = np.abs(flux_values - mean_flux) > 5 * std_flux
            
            if np.sum(outliers) > len(flux_values) * 0.1:
                return 0.3  # Poor quality due to many outliers
            
            return 1.0  # Good quality
            
        except Exception as e:
            self.logger.error(f"Error detecting systematic errors: {e}")
            return 0.5
    
    def _calculate_overall_score(self, quality_scores):
        """Calculate overall quality score."""
        try:
            weights = {
                'completeness': 0.3,
                'noise_level': 0.3,
                'max_gap': 0.2,
                'systematic_errors': 0.2
            }
            
            overall_score = sum(
                quality_scores.get(metric, 0) * weight
                for metric, weight in weights.items()
            )
            
            return min(overall_score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            self.logger.error(f"Error calculating overall score: {e}")
            return 0.5 