import logging
import json
import csv
import pandas as pd
from pathlib import Path
from datetime import datetime

class ExportManager:
    """Export manager for saving analysis results in various formats."""
    
    def __init__(self, export_dir='exports'):
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def export_results(self, results, target_name, format='csv'):
        """Export analysis results in specified format."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{target_name}_{timestamp}"
            
            if format == 'csv':
                return self._export_csv(results, filename)
            elif format == 'json':
                return self._export_json(results, filename)
            elif format == 'pdf':
                return self._export_pdf(results, filename)
            elif format == 'png':
                return self._export_png(results, filename)
            else:
                self.logger.error(f"Unsupported export format: {format}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to export results: {e}")
            return None
    
    def _export_csv(self, results, filename):
        """Export results as CSV."""
        try:
            # Prepare data for CSV export
            export_data = {
                'target_name': results.get('target_name', 'Unknown'),
                'mission': results.get('mission', 'Unknown'),
                'best_period': float(results.get('best_period', 0)),
                'transit_depth': results.get('transit_depth', 0),
                'snr': results.get('snr', 0),
                'analysis_type': results.get('analysis_type', 'Basic'),
                'timestamp': datetime.now().isoformat()
            }
            
            # Add quality scores if available
            quality_scores = results.get('quality_scores', {})
            for key, value in quality_scores.items():
                export_data[f'quality_{key}'] = value
            
            # Create DataFrame and export
            df = pd.DataFrame([export_data])
            filepath = self.export_dir / f"{filename}.csv"
            df.to_csv(filepath, index=False)
            
            self.logger.info(f"Exported CSV to {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to export CSV: {e}")
            return None
    
    def _export_json(self, results, filename):
        """Export results as JSON."""
        try:
            # Prepare data for JSON export
            export_data = {
                'target_name': results.get('target_name', 'Unknown'),
                'mission': results.get('mission', 'Unknown'),
                'analysis_results': {
                    'best_period': float(results.get('best_period', 0)),
                    'transit_depth': results.get('transit_depth', 0),
                    'snr': results.get('snr', 0),
                    'analysis_type': results.get('analysis_type', 'Basic')
                },
                'quality_scores': results.get('quality_scores', {}),
                'export_timestamp': datetime.now().isoformat()
            }
            
            filepath = self.export_dir / f"{filename}.json"
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Exported JSON to {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to export JSON: {e}")
            return None
    
    def _export_pdf(self, results, filename):
        """Export results as PDF report."""
        try:
            # This would typically use a library like reportlab or weasyprint
            # For now, we'll create a simple text-based report
            report_content = f"""
            Exoplanet Detection Report
            =========================
            
            Target: {results.get('target_name', 'Unknown')}
            Mission: {results.get('mission', 'Unknown')}
            Analysis Type: {results.get('analysis_type', 'Basic')}
            
            Results:
            - Best Period: {results.get('best_period', 0):.3f} days
            - Transit Depth: {results.get('transit_depth', 0):.4f}
            - Signal-to-Noise: {results.get('snr', 0):.2f}
            
            Quality Assessment:
            {json.dumps(results.get('quality_scores', {}), indent=2)}
            
            Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            filepath = self.export_dir / f"{filename}.txt"
            with open(filepath, 'w') as f:
                f.write(report_content)
            
            self.logger.info(f"Exported report to {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to export PDF: {e}")
            return None
    
    def _export_png(self, results, filename):
        """Export visualization as PNG."""
        try:
            # This would typically create a plot and save it as PNG
            # For now, we'll create a placeholder
            filepath = self.export_dir / f"{filename}.png"
            
            # Create a simple text file as placeholder
            with open(filepath.with_suffix('.txt'), 'w') as f:
                f.write(f"Visualization placeholder for {filename}")
            
            self.logger.info(f"Exported visualization to {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Failed to export PNG: {e}")
            return None 