import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

class DisplayManager:
    """Manager for displaying analysis results and visualizations."""
    
    def __init__(self):
        pass
    
    def display_overview(self, results):
        """Display analysis overview."""
        st.header("📊 Analysis Overview")
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Target", results.get('target_name', 'Unknown'))
        
        with col2:
            st.metric("Mission", results.get('mission', 'Unknown'))
        
        with col3:
            st.metric("Analysis Type", results.get('analysis_type', 'Basic'))
        
        # Detection metrics
        st.subheader("🔍 Detection Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            period = results.get('best_period', 0)
            if hasattr(period, 'value'):
                period = period.value
            st.metric("Best Period", f"{period:.3f} days")
        
        with col2:
            st.metric("Transit Depth", f"{results.get('transit_depth', 0):.4f}")
        
        with col3:
            st.metric("Signal-to-Noise", f"{results.get('snr', 0):.2f}")
    
    def display_light_curve(self, results):
        """Display light curve visualization."""
        st.subheader("📈 Light Curve")
        
        try:
            lc = results.get('lc')
            if lc is None:
                st.warning("No light curve data available")
                return
            
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
                title='Light Curve',
                xaxis_title='Time (days)',
                yaxis_title='Relative Flux',
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error displaying light curve: {e}")
    
    def display_periodogram(self, results):
        """Display periodogram visualization."""
        st.subheader("📊 Periodogram")
        
        try:
            bls = results.get('bls')
            if bls is None:
                st.warning("No periodogram data available")
                return
            
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
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error displaying periodogram: {e}")
    
    def display_phase_fold(self, results):
        """Display phase-folded light curve."""
        st.subheader("🔄 Phase-folded Light Curve")
        
        try:
            lc_folded = results.get('lc_folded')
            if lc_folded is None:
                st.warning("No phase-folded data available")
                return
            
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
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error displaying phase-folded light curve: {e}")
    
    def display_quality_assessment(self, results):
        """Display data quality assessment."""
        st.subheader("📈 Data Quality Assessment")
        
        quality_scores = results.get('quality_scores', {})
        if not quality_scores:
            st.warning("No quality assessment data available")
            return
        
        # Create quality metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Overall Score", f"{quality_scores.get('overall_score', 0):.2f}")
        
        with col2:
            st.metric("Completeness", f"{quality_scores.get('completeness', 0):.2f}")
        
        with col3:
            st.metric("Noise Level", f"{quality_scores.get('noise_level', 0):.4f}")
        
        with col4:
            st.metric("Max Gap", f"{quality_scores.get('max_gap', 0):.2f} days")
        
        # Quality assessment summary
        overall_score = quality_scores.get('overall_score', 0)
        
        if overall_score > 0.8:
            st.success("✅ High quality data - Excellent for analysis")
        elif overall_score > 0.6:
            st.info("🔍 Good quality data - Suitable for analysis")
        elif overall_score > 0.4:
            st.warning("⚠️ Moderate quality data - Use with caution")
        else:
            st.error("❌ Poor quality data - Consider alternative targets")
    
    def display_candidate_assessment(self, results):
        """Display exoplanet candidate assessment."""
        st.subheader("🌍 Exoplanet Candidate Assessment")
        
        # Calculate confidence score
        snr = results.get('snr', 0)
        transit_depth = results.get('transit_depth', 0)
        quality_score = results.get('quality_scores', {}).get('overall_score', 0)
        
        confidence = 0.0
        if snr > 7.0:
            confidence += 0.4
        if transit_depth > 0.001:
            confidence += 0.3
        if quality_score > 0.7:
            confidence += 0.3
        
        confidence = min(confidence, 1.0)
        
        # Display confidence
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Confidence Score", f"{confidence:.2f}")
        
        with col2:
            if confidence > 0.8:
                st.success("🎉 HIGH CONFIDENCE CANDIDATE")
            elif confidence > 0.6:
                st.info("🔍 MEDIUM CONFIDENCE CANDIDATE")
            elif confidence > 0.4:
                st.warning("⚠️ LOW CONFIDENCE CANDIDATE")
            else:
                st.error("❌ UNLIKELY CANDIDATE")
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        if confidence > 0.8:
            st.markdown("""
            - **Prioritize for follow-up observations**
            - **Submit for spectroscopic confirmation**
            - **Consider radial velocity measurements**
            - **Monitor for additional transits**
            """)
        elif confidence > 0.6:
            st.markdown("""
            - **Collect additional photometric data**
            - **Perform detailed false positive analysis**
            - **Consider ground-based follow-up**
            - **Monitor for transit timing variations**
            """)
        elif confidence > 0.4:
            st.markdown("""
            - **Requires additional data for confirmation**
            - **Perform comprehensive false positive analysis**
            - **Consider alternative explanations**
            - **Monitor for stellar activity**
            """)
        else:
            st.markdown("""
            - **Unlikely to be a real exoplanet**
            - **Consider stellar variability or instrumental effects**
            - **Try different analysis parameters**
            - **Consider alternative targets**
            """) 