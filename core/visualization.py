import logging
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

class VisualizationManager:
    """Visualization manager for creating publication-ready plots."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_publication_ready_plot(self, results, plot_type='transit'):
        """Create publication-ready plots."""
        try:
            if plot_type == 'transit':
                return self._create_transit_plot(results)
            elif plot_type == 'periodogram':
                return self._create_periodogram_plot(results)
            elif plot_type == 'phase_fold':
                return self._create_phase_fold_plot(results)
            else:
                return self._create_generic_plot(results)
                
        except Exception as e:
            self.logger.error(f"Failed to create {plot_type} plot: {e}")
            return None
    
    def create_multiple_plots(self, results, plot_types=None):
        """Create multiple plots for comprehensive visualization."""
        try:
            if plot_types is None:
                plot_types = ['transit', 'periodogram', 'phase_fold']
            
            plots = {}
            for plot_type in plot_types:
                plot = self.create_publication_ready_plot(results, plot_type)
                if plot is not None:
                    plots[plot_type] = plot
            
            self.logger.info("Plot generation completed")
            self.logger.info(f"Generated {len(plots)} plots")
            return plots
            
        except Exception as e:
            self.logger.error(f"Failed to create multiple plots: {e}")
            return {}
    
    def _create_transit_plot(self, results):
        """Create transit light curve plot."""
        try:
            lc = results.get('lc')
            if lc is None:
                return None
            
            # Extract time and flux data
            if hasattr(lc.time, 'value'):
                time_data = lc.time.value
            else:
                time_data = lc.time
            
            if hasattr(lc.flux, 'value'):
                flux_data = lc.flux.value
            else:
                flux_data = lc.flux
            
            # Create plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=time_data,
                y=flux_data,
                mode='lines',
                name='Light Curve',
                line=dict(color='blue', width=1)
            ))
            
            fig.update_layout(
                title='Light Curve - Transit Detection',
                xaxis_title='Time (days)',
                yaxis_title='Relative Flux',
                template='plotly_white',
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating transit plot: {e}")
            return None
    
    def _create_periodogram_plot(self, results):
        """Create periodogram plot."""
        try:
            bls = results.get('bls')
            if bls is None:
                return None
            
            # Extract period and power data
            if hasattr(bls.period, 'value'):
                period_data = bls.period.value
            else:
                period_data = bls.period
            
            if hasattr(bls.power, 'value'):
                power_data = bls.power.value
            else:
                power_data = bls.power
            
            # Create plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=period_data,
                y=power_data,
                mode='lines',
                name='BLS Periodogram',
                line=dict(color='red', width=2)
            ))
            
            # Mark the best period
            best_period = results.get('best_period')
            if best_period is not None:
                if hasattr(best_period, 'value'):
                    best_period_val = best_period.value
                else:
                    best_period_val = best_period
                
                fig.add_vline(
                    x=best_period_val,
                    line_dash="dash",
                    line_color="green",
                    annotation_text=f"Best Period: {best_period_val:.3f} days"
                )
            
            fig.update_layout(
                title='BLS Periodogram',
                xaxis_title='Period (days)',
                yaxis_title='Power',
                template='plotly_white',
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating periodogram plot: {e}")
            return None
    
    def _create_phase_fold_plot(self, results):
        """Create phase-folded light curve plot."""
        try:
            lc_folded = results.get('lc_folded')
            if lc_folded is None:
                return None
            
            # Extract phase and flux data
            if hasattr(lc_folded.phase, 'value'):
                phase_data = lc_folded.phase.value
            else:
                phase_data = lc_folded.phase
            
            if hasattr(lc_folded.flux, 'value'):
                flux_data = lc_folded.flux.value
            else:
                flux_data = lc_folded.flux
            
            # Create plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=phase_data,
                y=flux_data,
                mode='markers',
                name='Phase-folded Light Curve',
                marker=dict(color='purple', size=3)
            ))
            
            fig.update_layout(
                title='Phase-folded Light Curve',
                xaxis_title='Phase',
                yaxis_title='Relative Flux',
                template='plotly_white',
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating phase fold plot: {e}")
            return None
    
    def _create_generic_plot(self, results):
        """Create a generic plot for any data."""
        try:
            # Create a simple scatter plot of available data
            fig = go.Figure()
            
            # Try to find any numeric data to plot
            for key, value in results.items():
                if isinstance(value, (list, np.ndarray)) and len(value) > 0:
                    if hasattr(value, 'value'):
                        data = value.value
                    else:
                        data = value
                    
                    if len(data) > 1:
                        fig.add_trace(go.Scatter(
                            y=data,
                            mode='lines',
                            name=key
                        ))
            
            fig.update_layout(
                title='Data Visualization',
                template='plotly_white',
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating generic plot: {e}")
            return None 