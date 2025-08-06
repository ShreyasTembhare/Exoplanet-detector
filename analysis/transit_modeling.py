import logging
import numpy as np
from scipy.optimize import curve_fit

class TransitModeler:
    """Transit modeling and fitting."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def model_transit(self, results, target_params=None):
        """Model transit using analytical or numerical methods."""
        try:
            # Extract data
            lc_folded = results.get('lc_folded')
            if lc_folded is None:
                return results
            
            # Get phase and flux data
            if hasattr(lc_folded.phase, 'value'):
                phase = lc_folded.phase.value
            else:
                phase = lc_folded.phase
            
            if hasattr(lc_folded.flux, 'value'):
                flux = lc_folded.flux.value
            else:
                flux = lc_folded.flux
            
            # Fit transit model
            transit_params = self._fit_transit_model(phase, flux, target_params)
            
            # Add modeling results
            results['transit_model'] = {
                'parameters': transit_params,
                'fitted_curve': self._generate_transit_curve(phase, transit_params),
                'goodness_of_fit': self._calculate_goodness_of_fit(flux, transit_params)
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Transit modeling failed: {e}")
            return results
    
    def _fit_transit_model(self, phase, flux, target_params=None):
        """Fit transit model to data."""
        try:
            # Simple transit model function
            def transit_model(phase, rp_rs, a_rs, t0):
                """Simple transit model."""
                # This is a simplified transit model
                # In practice, you'd use a more sophisticated model like batman
                transit_signal = np.ones_like(phase)
                
                # Find transit regions
                transit_mask = np.abs(phase - t0) < 0.1
                transit_signal[transit_mask] = 1.0 - rp_rs**2
                
                return transit_signal
            
            # Initial parameter guesses
            if target_params is None:
                p0 = [0.1, 10.0, 0.5]  # rp_rs, a_rs, t0
            else:
                p0 = [
                    target_params.get('rp_rs', 0.1),
                    target_params.get('a_rs', 10.0),
                    target_params.get('t0', 0.5)
                ]
            
            # Fit the model
            popt, pcov = curve_fit(transit_model, phase, flux, p0=p0, maxfev=1000)
            
            return {
                'rp_rs': popt[0],  # Planet-to-star radius ratio
                'a_rs': popt[1],   # Semi-major axis to star radius ratio
                't0': popt[2],     # Transit center
                'uncertainties': np.sqrt(np.diag(pcov))
            }
            
        except Exception as e:
            self.logger.error(f"Transit model fitting failed: {e}")
            return {
                'rp_rs': 0.1,
                'a_rs': 10.0,
                't0': 0.5,
                'uncertainties': [0.01, 1.0, 0.01]
            }
    
    def _generate_transit_curve(self, phase, params):
        """Generate transit curve from fitted parameters."""
        try:
            def transit_model(phase, rp_rs, a_rs, t0):
                transit_signal = np.ones_like(phase)
                transit_mask = np.abs(phase - t0) < 0.1
                transit_signal[transit_mask] = 1.0 - rp_rs**2
                return transit_signal
            
            return transit_model(phase, params['rp_rs'], params['a_rs'], params['t0'])
            
        except Exception as e:
            self.logger.error(f"Failed to generate transit curve: {e}")
            return np.ones_like(phase)
    
    def _calculate_goodness_of_fit(self, observed, params):
        """Calculate goodness of fit metrics."""
        try:
            # Generate model prediction
            phase = np.linspace(0, 1, len(observed))
            predicted = self._generate_transit_curve(phase, params)
            
            # Calculate residuals
            residuals = observed - predicted
            
            # Calculate metrics
            chi_squared = np.sum(residuals**2)
            r_squared = 1 - np.sum(residuals**2) / np.sum((observed - np.mean(observed))**2)
            rmse = np.sqrt(np.mean(residuals**2))
            
            return {
                'chi_squared': chi_squared,
                'r_squared': r_squared,
                'rmse': rmse,
                'residuals': residuals
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate goodness of fit: {e}")
            return {
                'chi_squared': float('inf'),
                'r_squared': 0.0,
                'rmse': float('inf'),
                'residuals': np.zeros_like(observed)
            } 