import logging
import numpy as np
import lightkurve as lk
from astropy import units as u
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from .ml_predictor import MLPredictor

class ExoplanetAnalyzer:
    """Main exoplanet analysis engine."""
    
    def __init__(self, resilient_processor, quality_assessor, cache_manager, db, ml_manager=None):
        self.resilient_processor = resilient_processor
        self.quality_assessor = quality_assessor
        self.cache_manager = cache_manager
        self.db = db
        self.ml_manager = ml_manager
        self.ml_predictor = MLPredictor(ml_manager) if ml_manager else None
        self.logger = logging.getLogger(__name__)
    
    def perform_analysis(self, target_name, mission="Kepler", analysis_type="Basic", 
                        quality="good", period_range=(0.5, 50), bin_size=1, 
                        detrend_method="flatten", significance_threshold=0.01):
        """Perform comprehensive exoplanet analysis."""
        try:
            self.logger.info(f"Starting analysis for {target_name} from {mission}")
            
            # Check cache first
            cache_key = f"{target_name}_{mission}_{analysis_type}"
            cached_result = self.cache_manager.get_cached_result(cache_key)
            if cached_result:
                self.logger.info("Using cached result")
                return cached_result
            
            # Data acquisition
            lc = self._acquire_data(target_name, mission, quality)
            if lc is None:
                raise Exception(f"No data found for {target_name}")
            
            # Data preprocessing
            lc_clean = self._preprocess_data(lc, detrend_method, bin_size)
            
            # Quality assessment
            quality_scores = self.quality_assessor.assess_light_curve_quality(lc_clean)
            
            # Periodogram analysis
            bls, ls = self._perform_periodogram_analysis(lc_clean, period_range)
            best_period = bls.period_at_max_power
            
            # Phase folding
            lc_folded = lc_clean.fold(period=best_period)
            
            # Calculate metrics
            try:
                # Handle units properly
                if hasattr(lc_folded.flux, 'value'):
                    flux_values = lc_folded.flux.value
                else:
                    flux_values = lc_folded.flux
                
                if hasattr(lc_clean.flux, 'value'):
                    clean_flux_values = lc_clean.flux.value
                else:
                    clean_flux_values = lc_clean.flux
                
                transit_depth = 1.0 - np.min(flux_values)
                snr = transit_depth / np.std(clean_flux_values)
                
            except Exception as e:
                self.logger.warning(f"Metric calculation failed: {e}")
                transit_depth = 0.0
                snr = 0.0
            
            # Get BLS power safely
            try:
                if hasattr(bls, 'power_at_max_power'):
                    bls_power = bls.power_at_max_power
                else:
                    bls_power = 0.0
            except:
                bls_power = 0.0
            
            # Compile results
            results = {
                'lc': lc,
                'lc_clean': lc_clean,
                'bls': bls,
                'ls': ls,
                'best_period': best_period,
                'lc_folded': lc_folded,
                'transit_depth': transit_depth,
                'snr': snr,
                'bls_power': bls_power,
                'quality_scores': quality_scores,
                'analysis_type': analysis_type,
                'mission': mission,
                'target_name': target_name
            }
            
            # Advanced analysis based on type
            if analysis_type in ["Advanced", "Comprehensive"]:
                results = self._apply_advanced_analysis(results)
            
            # Cache results
            self.cache_manager.cache_result(cache_key, results)
            
            # Save to database
            self.db.save_analysis_result(target_name, mission, results)
            
            self.logger.info(f"Analysis completed for {target_name}")
            return results
            
        except Exception as e:
            self.logger.error(f"Analysis failed for {target_name}: {e}")
            raise
    
    def _acquire_data(self, target_name, mission, quality):
        """Acquire light curve data."""
        try:
            if target_name == "DEMO_TARGET":
                return self._create_demo_lightcurve()
            
            # Try multiple quarters/missions with fallback
            quarters_to_try = [16, 15, 14, 13, 12]
            
            for quarter in quarters_to_try:
                try:
                    if mission == "Kepler":
                        search_result = lk.search_targetpixelfile(target_name, quarter=quarter)
                    elif mission == "TESS":
                        search_result = lk.search_tesscut(target_name)
                    else:
                        search_result = lk.search_targetpixelfile(target_name, quarter=quarter)
                    
                    if len(search_result) > 0:
                        tpf = search_result[0].download(quality=quality)
                        lc = tpf.to_lightcurve(aperture_mask='pipeline')
                        return lc
                except:
                    continue
            
            # Use fallback data acquisition
            return self.resilient_processor.execute_with_retry(
                self.resilient_processor._fallback_data_download, target_name, mission
            )
            
        except Exception as e:
            self.logger.error(f"Data acquisition failed: {e}")
            return None
    
    def _preprocess_data(self, lc, detrend_method, bin_size):
        """Preprocess light curve data."""
        try:
            # Detrending
            if detrend_method == "flatten":
                lc_clean = lc.flatten(window_length=101)
            elif detrend_method == "spline":
                lc_clean = lc.flatten(window_length=101, polyorder=3)
            elif detrend_method == "polynomial":
                lc_clean = lc.flatten(window_length=101, polyorder=2)
            else:
                lc_clean = lc
            
            # Binning
            if bin_size > 1:
                lc_clean = lc_clean.bin(binsize=bin_size)
            
            return lc_clean
            
        except Exception as e:
            self.logger.error(f"Data preprocessing failed: {e}")
            return lc
    
    def _perform_periodogram_analysis(self, lc_clean, period_range):
        """Perform periodogram analysis."""
        try:
            # BLS periodogram
            bls = lc_clean.to_periodogram(
                method='bls', 
                minimum_period=period_range[0], 
                maximum_period=period_range[1]
            )
            
            # Lomb-Scargle periodogram
            ls = lc_clean.to_periodogram(
                method='lombscargle', 
                minimum_period=period_range[0], 
                maximum_period=period_range[1]
            )
            
            return bls, ls
            
        except Exception as e:
            self.logger.error(f"Periodogram analysis failed: {e}")
            raise
    
    def _apply_advanced_analysis(self, results):
        """Apply advanced analysis techniques."""
        try:
            # This would integrate with other analysis modules
            # For now, we'll add placeholder advanced features
            
            # Add confidence score
            snr = results.get('snr', 0)
            transit_depth = results.get('transit_depth', 0)
            
            confidence = 0.0
            if snr > 7.0:
                confidence += 0.4
            if transit_depth > 0.001:
                confidence += 0.3
            if results.get('quality_scores', {}).get('overall_score', 0) > 0.7:
                confidence += 0.3
            
            results['confidence_score'] = min(confidence, 1.0)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Advanced analysis failed: {e}")
            return results
    
    def _create_demo_lightcurve(self):
        """Create a demo light curve for testing."""
        class DemoLightCurve:
            def __init__(self):
                self.time = np.linspace(0, 100, 10000)
                period = 3.5
                transit_depth = 0.01
                transit_duration = 0.1
                phase = (self.time % period) / period
                transit_center = 0.5
                transit_width = transit_duration / (24 * period)
                transit_signal = 1.0 - transit_depth * np.exp(-0.5 * ((phase - transit_center) / transit_width)**2)
                stellar_variability = 0.005 * np.sin(2 * np.pi * self.time / 25)
                noise = 0.002 * np.random.normal(0, 1, len(self.time))
                self.flux = transit_signal + stellar_variability + noise
                self.time = self.time * u.day
                self.flux = self.flux * u.dimensionless_unscaled
            
            def flatten(self, window_length=101):
                return self
            
            def to_periodogram(self, method='bls', minimum_period=0.5, maximum_period=50):
                class DemoPeriodogram:
                    def __init__(self):
                        self.period_at_max_power = 3.5 * u.day
                        self.power_at_max_power = 0.15
                return DemoPeriodogram()
            
            def fold(self, period=None):
                if period is None:
                    period = 3.5
                phase = (self.time.value % period) / period
                return DemoFolded(phase, self.flux.value)
            
            def bin(self, binsize=1):
                return self
        
        class DemoFolded:
            def __init__(self, phase, flux):
                self.phase = phase * u.dimensionless_unscaled
                self.flux = flux * u.dimensionless_unscaled
        
        return DemoLightCurve() 