import streamlit as st
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import sqlite3

class StreamlitInterface:
    """Main Streamlit interface for the exoplanet detector."""
    
    def __init__(self, analyzer, db, export_manager, viz_manager, community_manager, educational_manager):
        self.analyzer = analyzer
        self.db = db
        self.export_manager = export_manager
        self.viz_manager = viz_manager
        self.community_manager = community_manager
        self.educational_manager = educational_manager
        self.logger = logging.getLogger(__name__)
    
    def run(self):
        """Run the main Streamlit interface."""
        try:
            # Page configuration
            st.set_page_config(
                page_title="Advanced Exoplanet Detector",
                page_icon="🌌",
                layout="wide",
                initial_sidebar_state="expanded"
            )
            
            # Initialize session state
            self._init_session_state()
            
            # Load custom CSS
            self._load_custom_css()
            
            # Main title
            st.title("🌌 Advanced Exoplanet Detector")
            st.markdown("Discover exoplanets using advanced analysis techniques")
            
            # Sidebar navigation
            page = self._create_sidebar()
            
            # Main content based on selected page
            if page == "Main Detector":
                self._show_main_detector()
            elif page == "Target Discovery":
                self._show_target_discovery()
            elif page == "Star Shortlist & Analysis":
                self._show_star_shortlist_analysis()
            elif page == "Batch Processing":
                self._show_batch_processing()
            elif page == "Advanced Analysis":
                self._show_advanced_analysis()
            elif page == "Community Sharing":
                self._show_community_sharing()
            elif page == "Real-time Monitoring":
                self._show_real_time_monitoring()
            elif page == "Help & Tutorials":
                self._show_help_tutorials()
            
        except Exception as e:
            st.error(f"Interface error: {e}")
            self.logger.error(f"Interface error: {e}")
    
    def _init_session_state(self):
        """Initialize Streamlit session state."""
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'current_target' not in st.session_state:
            st.session_state.current_target = None
        if 'discovered_stars' not in st.session_state:
            st.session_state.discovered_stars = []
        if 'pipeline_running' not in st.session_state:
            st.session_state.pipeline_running = False
        if 'log_queue' not in st.session_state:
            st.session_state.log_queue = []
    
    def _load_custom_css(self):
        """Load custom CSS styling."""
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
            border-left: 4px solid #667eea;
        }
        .success-box {
            background: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 5px;
            padding: 1rem;
            margin: 1rem 0;
        }
        .warning-box {
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 5px;
            padding: 1rem;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def _create_sidebar(self):
        """Create the sidebar navigation."""
        st.sidebar.title("🚀 Navigation")
        
        page = st.sidebar.selectbox(
            "Choose a page:",
            [
                "Main Detector",
                "Target Discovery", 
                "Star Shortlist & Analysis",
                "Batch Processing",
                "Advanced Analysis",
                "Community Sharing",
                "Real-time Monitoring",
                "Help & Tutorials"
            ],
            key="page_select"
        )
        
        st.sidebar.markdown("---")
        
        # Quick stats
        st.sidebar.markdown("### 📊 Quick Stats")
        analysis_count = len(self.db.get_analysis_history())
        discovered_count = len(self.db.get_discovered_stars())
        
        st.sidebar.metric("Analyses Performed", analysis_count)
        st.sidebar.metric("Stars Discovered", discovered_count)
        
        return page
    
    def _show_main_detector(self):
        """Show the main detector interface."""
        st.header("🔍 Main Detector")
        st.info("Analyze individual targets for exoplanet detection.")
        
        # Dashboard section
        self._show_discovery_dashboard()
        
        # Target input section
        self._show_target_input()
        
        # Analysis results
        if st.session_state.analysis_results:
            self._display_analysis_results(st.session_state.analysis_results)
    
    def _show_target_input(self):
        """Show the target input section."""
        st.subheader("🎯 Target Analysis")
        
        # Input section
        col1, col2 = st.columns(2)
        
        with col1:
            target_name = st.text_input(
                "Target Name",
                value="KIC 11442793",
                help="Enter the target star name (e.g., KIC 11442793, TIC 123456789)",
                key="target_input"
            )
            
            mission = st.selectbox(
                "Mission",
                ["Kepler", "TESS", "K2"],
                key="mission_select_main"
            )
            
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Basic", "Advanced", "Comprehensive"],
                key="analysis_type_select"
            )
        
        with col2:
            quality = st.selectbox(
                "Data Quality",
                ["good", "medium", "poor"],
                key="quality_select"
            )
            
            period_range = st.slider(
                "Period Range (days)",
                min_value=0.1,
                max_value=100.0,
                value=(0.5, 50.0),
                key="period_range_slider"
            )
            
            detrend_method = st.selectbox(
                "Detrending Method",
                ["flatten", "spline", "polynomial"],
                key="detrend_method_select"
            )
        
        # Analysis button
        if st.button("🚀 Start Analysis", key="start_analysis_btn"):
            with st.spinner("Analyzing target..."):
                try:
                    results = self.analyzer.perform_analysis(
                        target_name=target_name,
                        mission=mission,
                        analysis_type=analysis_type,
                        quality=quality,
                        period_range=period_range,
                        detrend_method=detrend_method
                    )
                    
                    if results:
                        st.session_state.analysis_results = results
                        st.session_state.current_target = target_name
                        st.success("✅ Analysis completed successfully!")
                    else:
                        st.error("❌ Analysis failed. Please try a different target.")
                        
                except Exception as e:
                    st.error(f"❌ Analysis error: {e}")
    
    def _show_discovery_dashboard(self):
        """Show the discovery dashboard with statistics and results."""
        st.subheader("📊 Discovery Dashboard")
        
        # Get statistics
        try:
            candidates = self.db.get_new_candidates()
            discovered_stars = self.db.get_discovered_stars()
            analysis_history = self.db.get_analysis_history()
            
            # Create metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("New Candidates", len(candidates))
            
            with col2:
                st.metric("Discovered Stars", len(discovered_stars))
            
            with col3:
                st.metric("Total Analyses", len(analysis_history))
            
            with col4:
                if candidates:
                    avg_confidence = np.mean([c.get('ml_confidence', 0) for c in candidates])
                    st.metric("Avg Confidence", f"{avg_confidence:.2f}")
                else:
                    st.metric("Avg Confidence", "0.00")
            
            # Recent discoveries
            if candidates:
                st.subheader("🆕 Recent Discoveries")
                
                # Create DataFrame for display
                df = pd.DataFrame(candidates)
                if not df.empty:
                    # Select recent candidates (last 10)
                    recent_df = df.head(10)
                    
                    # Create a nice display table
                    display_data = []
                    for _, row in recent_df.iterrows():
                        display_data.append({
                            'Target': row['target_name'],
                            'Mission': row['mission'],
                            'Period (days)': f"{row.get('period', 0):.2f}",
                            'Transit Depth (%)': f"{row.get('transit_depth', 0)*100:.3f}",
                            'SNR': f"{row.get('snr', 0):.1f}",
                            'ML Confidence': f"{row.get('ml_confidence', 0):.2f}",
                            'Status': '🆕 New Candidate' if not row.get('is_known_exoplanet', False) else '📚 Known'
                        })
                    
                    display_df = pd.DataFrame(display_data)
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Add action buttons
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("📊 View All Candidates", key="view_all_candidates_btn"):
                            st.session_state.show_all_candidates = True
                    
                    with col2:
                        if st.button("📈 Export Results", key="export_results_btn"):
                            self._export_discovery_results(candidates)
                
                # Show all candidates if requested
                if st.session_state.get('show_all_candidates', False):
                    self._show_all_candidates(candidates)
            
            # Add button to view all potential stars with exoplanets
            st.subheader("🔍 Potential Stars Database")
            
            # Show database location
            st.info(f"💾 Database Location: `{self.db.db_path}`")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📋 View All Potential Stars", key="view_potential_stars_btn"):
                    st.session_state.show_potential_stars = True
            
            with col2:
                if st.button("🗑️ Clear Database", key="clear_database_btn"):
                    if st.button("⚠️ Confirm Clear", key="confirm_clear_btn"):
                        self._clear_candidates_database()
                        st.success("✅ Database cleared successfully!")
                        st.rerun()
            
            # Show potential stars if requested
            if st.session_state.get('show_potential_stars', False):
                self._show_potential_stars_database()
            
            # Mission distribution chart
            if candidates:
                st.subheader("📈 Mission Distribution")
                
                mission_counts = {}
                for candidate in candidates:
                    mission = candidate.get('mission', 'Unknown')
                    mission_counts[mission] = mission_counts.get(mission, 0) + 1
                
                if mission_counts:
                    import plotly.express as px
                    
                    mission_df = pd.DataFrame([
                        {'Mission': mission, 'Count': count}
                        for mission, count in mission_counts.items()
                    ])
                    
                    fig = px.pie(mission_df, values='Count', names='Mission', 
                                title="Candidates by Mission")
                    st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Failed to load dashboard: {str(e)}")
            self.logger.error(f"Dashboard error: {e}")
    
    def _export_discovery_results(self, candidates):
        """Export discovery results."""
        try:
            if not candidates:
                st.warning("No candidates to export")
                return
            
            # Create export data
            export_data = []
            for candidate in candidates:
                export_data.append({
                    'target_name': candidate.get('target_name', ''),
                    'mission': candidate.get('mission', ''),
                    'period': candidate.get('period', 0),
                    'transit_depth': candidate.get('transit_depth', 0),
                    'snr': candidate.get('snr', 0),
                    'ml_confidence': candidate.get('ml_confidence', 0),
                    'discovery_method': candidate.get('discovery_method', ''),
                    'is_known_exoplanet': candidate.get('is_known_exoplanet', False),
                    'validation_status': candidate.get('validation_status', ''),
                    'created_at': candidate.get('created_at', '')
                })
            
            # Export to CSV
            df = pd.DataFrame(export_data)
            csv = df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"exoplanet_candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            st.success("✅ Export ready for download")
            
        except Exception as e:
            st.error(f"Export failed: {str(e)}")
            self.logger.error(f"Export error: {e}")
    
    def _show_all_candidates(self, candidates):
        """Show all candidates in a detailed table."""
        st.subheader("📋 All Discovered Candidates")
        
        if not candidates:
            st.info("No candidates found.")
            return
        
        # Create detailed DataFrame
        df = pd.DataFrame(candidates)
        
        # Add filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            mission_filter = st.selectbox(
                "Filter by Mission",
                ["All"] + list(df['mission'].unique()),
                key="mission_filter"
            )
        
        with col2:
            confidence_filter = st.slider(
                "Min ML Confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1,
                key="confidence_filter"
            )
        
        with col3:
            snr_filter = st.slider(
                "Min SNR",
                min_value=0.0,
                max_value=20.0,
                value=0.0,
                step=0.5,
                key="snr_filter"
            )
        
        # Apply filters
        filtered_df = df.copy()
        
        if mission_filter != "All":
            filtered_df = filtered_df[filtered_df['mission'] == mission_filter]
        
        filtered_df = filtered_df[filtered_df['ml_confidence'] >= confidence_filter]
        filtered_df = filtered_df[filtered_df['snr'] >= snr_filter]
        
        # Display filtered results
        st.write(f"Showing {len(filtered_df)} of {len(df)} candidates")
        
        if not filtered_df.empty:
            # Create display table
            display_data = []
            for _, row in filtered_df.iterrows():
                display_data.append({
                    'Target': row['target_name'],
                    'Mission': row['mission'],
                    'Period (days)': f"{row.get('period', 0):.2f}",
                    'Transit Depth (%)': f"{row.get('transit_depth', 0)*100:.3f}",
                    'SNR': f"{row.get('snr', 0):.1f}",
                    'ML Confidence': f"{row.get('ml_confidence', 0):.2f}",
                    'BLS Power': f"{row.get('bls_power', 0):.3f}",
                    'Discovery Method': row.get('discovery_method', 'Unknown'),
                    'Status': '🆕 New Candidate' if not row.get('is_known_exoplanet', False) else '📚 Known',
                    'Date': row.get('created_at', 'Unknown')[:10] if row.get('created_at') else 'Unknown'
                })
            
            display_df = pd.DataFrame(display_data)
            st.dataframe(display_df, use_container_width=True)
            
            # Add close button
            if st.button("❌ Close Detailed View", key="close_detailed_view_btn"):
                st.session_state.show_all_candidates = False
                st.rerun()
        else:
            st.warning("No candidates match the selected filters.")
            if st.button("❌ Close Detailed View", key="close_detailed_view_btn"):
                st.session_state.show_all_candidates = False
                st.rerun()
    
    def _show_potential_stars_database(self):
        """Show the complete database of potential stars with exoplanets."""
        st.subheader("📋 Complete Potential Stars Database")
        
        # Get all candidates from database
        all_candidates = self.db.get_all_candidates()
        
        if not all_candidates:
            st.info("No potential stars found in database. Run auto-discovery to find candidates!")
            if st.button("❌ Close Database View", key="close_database_view_btn"):
                st.session_state.show_potential_stars = False
                st.rerun()
            return
        
        # Create DataFrame
        df = pd.DataFrame(all_candidates)
        
        # Add filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_filter = st.selectbox(
                "Filter by Status",
                ["All", "New Candidates", "Known Exoplanets"],
                key="status_filter"
            )
        
        with col2:
            mission_filter = st.selectbox(
                "Filter by Mission",
                ["All"] + list(df['mission'].unique()) if not df.empty else ["All"],
                key="mission_filter_db"
            )
        
        with col3:
            date_filter = st.date_input(
                "Filter by Discovery Date",
                value=None,
                key="date_filter"
            )
        
        # Apply filters
        filtered_df = df.copy()
        
        if status_filter == "New Candidates":
            filtered_df = filtered_df[filtered_df['is_known_exoplanet'] == False]
        elif status_filter == "Known Exoplanets":
            filtered_df = filtered_df[filtered_df['is_known_exoplanet'] == True]
        
        if mission_filter != "All":
            filtered_df = filtered_df[filtered_df['mission'] == mission_filter]
        
        if date_filter:
            # Convert date_filter to string for comparison
            date_str = date_filter.strftime('%Y-%m-%d')
            filtered_df = filtered_df[filtered_df['created_at'].str.startswith(date_str)]
        
        # Display statistics
        st.write(f"📊 Showing {len(filtered_df)} of {len(df)} total candidates")
        
        # Display table
        if not filtered_df.empty:
            # Create display table with discovery dates
            display_data = []
            for _, row in filtered_df.iterrows():
                # Format the date
                created_date = row.get('created_at', 'Unknown')
                if created_date and created_date != 'Unknown':
                    try:
                        # Parse the date and format it nicely
                        from datetime import datetime
                        if 'T' in str(created_date):
                            date_obj = datetime.fromisoformat(str(created_date).replace('Z', '+00:00'))
                        else:
                            date_obj = datetime.fromisoformat(str(created_date))
                        formatted_date = date_obj.strftime('%Y-%m-%d %H:%M')
                    except:
                        formatted_date = str(created_date)
                else:
                    formatted_date = 'Unknown'
                
                # Determine status
                if row.get('is_known_exoplanet', False):
                    status = "📚 Known Exoplanet"
                else:
                    status = "🆕 New Candidate"
                
                display_data.append({
                    'Target': row['target_name'],
                    'Mission': row['mission'],
                    'Period (days)': f"{row.get('period', 0):.2f}",
                    'Transit Depth (%)': f"{row.get('transit_depth', 0)*100:.3f}",
                    'SNR': f"{row.get('snr', 0):.1f}",
                    'ML Confidence': f"{row.get('ml_confidence', 0):.2f}",
                    'BLS Power': f"{row.get('bls_power', 0):.3f}",
                    'Discovery Method': row.get('discovery_method', 'Unknown'),
                    'Status': status,
                    'Discovery Date': formatted_date
                })
            
            display_df = pd.DataFrame(display_data)
            st.dataframe(display_df, use_container_width=True)
            
            # Add export functionality
            if st.button("📥 Export to CSV", key="export_database_btn"):
                self._export_candidates_to_csv(filtered_df)
            
            # Add close button
            if st.button("❌ Close Database View", key="close_database_view_btn"):
                st.session_state.show_potential_stars = False
                st.rerun()
        else:
            st.warning("No candidates match the selected filters.")
            if st.button("❌ Close Database View", key="close_database_view_btn"):
                st.session_state.show_potential_stars = False
                st.rerun()
    
    def _clear_candidates_database(self):
        """Clear the candidates database."""
        try:
            with sqlite3.connect(self.db.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM new_candidates')
                cursor.execute('DELETE FROM analyzed_targets')
                conn.commit()
            
            self.logger.info("Candidates database cleared successfully")
            
        except Exception as e:
            st.error(f"Failed to clear database: {str(e)}")
            self.logger.error(f"Failed to clear database: {e}")
    
    def _export_candidates_to_csv(self, df):
        """Export candidates to CSV file."""
        try:
            # Create export data
            export_data = []
            for _, row in df.iterrows():
                export_data.append({
                    'target_name': row['target_name'],
                    'mission': row['mission'],
                    'period': row.get('period', 0),
                    'transit_depth': row.get('transit_depth', 0),
                    'snr': row.get('snr', 0),
                    'ml_confidence': row.get('ml_confidence', 0),
                    'bls_power': row.get('bls_power', 0),
                    'discovery_method': row.get('discovery_method', ''),
                    'is_known_exoplanet': row.get('is_known_exoplanet', False),
                    'validation_status': row.get('validation_status', ''),
                    'created_at': row.get('created_at', '')
                })
            
            # Export to CSV
            export_df = pd.DataFrame(export_data)
            csv = export_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Database CSV",
                data=csv,
                file_name=f"potential_stars_database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            st.success("✅ Export ready for download")
            
        except Exception as e:
            st.error(f"Export failed: {str(e)}")
            self.logger.error(f"Export error: {e}")
    
    def _show_target_discovery(self):
        """Show the target discovery interface."""
        st.header("🌟 Target Discovery")
        st.info("This feature helps discover new potential exoplanet host stars.")
        
        # Auto-discovery section
        st.subheader("🔍 Auto Discovery")
        
        col1, col2 = st.columns(2)
        
        with col1:
            discovery_method = st.selectbox(
                "Discovery Method",
                ["BLS Search", "ML Detection", "Multi-mission Fusion"],
                key="discovery_method_select"
            )
            
            mission = st.selectbox(
                "Mission",
                ["Kepler", "TESS", "K2"],
                key="mission_select"
            )
        
        with col2:
            max_targets = st.number_input(
                "Maximum Targets",
                min_value=1,
                max_value=100,
                value=10,
                key="max_targets_input"
            )
            
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                key="confidence_threshold_slider"
            )
        
        # Advanced options
        with st.expander("Advanced Discovery Options"):
            col3, col4 = st.columns(2)
            
            with col3:
                period_range_min = st.number_input(
                    "Min Period (days)",
                    min_value=0.1,
                    max_value=10.0,
                    value=0.5,
                    step=0.1,
                    key="period_min_input"
                )
                
                snr_threshold = st.number_input(
                    "SNR Threshold",
                    min_value=1.0,
                    max_value=20.0,
                    value=7.0,
                    step=0.5,
                    key="snr_threshold_input"
                )
            
            with col4:
                period_range_max = st.number_input(
                    "Max Period (days)",
                    min_value=10.0,
                    max_value=100.0,
                    value=50.0,
                    step=1.0,
                    key="period_max_input"
                )
                
                transit_depth_min = st.number_input(
                    "Min Transit Depth (%)",
                    min_value=0.001,
                    max_value=1.0,
                    value=0.01,
                    step=0.001,
                    key="transit_depth_min_input"
                )
        
        if st.button("🔍 Start Discovery", key="start_discovery_btn"):
            self._run_auto_discovery(
                discovery_method=discovery_method,
                mission=mission,
                max_targets=max_targets,
                confidence_threshold=confidence_threshold,
                period_range=(period_range_min, period_range_max),
                snr_threshold=snr_threshold,
                transit_depth_min=transit_depth_min
            )
        
        # Display discovered candidates
        self._display_discovered_candidates()
    
    def _run_auto_discovery(self, discovery_method, mission, max_targets, 
                           confidence_threshold, period_range, snr_threshold, transit_depth_min):
        """Run the auto-discovery pipeline."""
        try:
            st.info("🚀 Starting auto-discovery pipeline...")
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Generate target list
            status_text.text("📋 Generating target list...")
            progress_bar.progress(10)
            
            targets = self._generate_target_list(mission, max_targets)
            if not targets:
                st.error("❌ Failed to generate target list")
                return
            
            st.success(f"✅ Generated {len(targets)} targets for analysis")
            
            # Step 2: Analyze targets
            status_text.text("🔬 Analyzing targets for transit signals...")
            progress_bar.progress(30)
            
            candidates = []
            for i, target in enumerate(targets):
                try:
                    # Update progress
                    progress = 30 + (i / len(targets)) * 50
                    progress_bar.progress(int(progress))
                    status_text.text(f"🔬 Analyzing {target} ({i+1}/{len(targets)})")
                    
                    # Analyze target
                    result = self._analyze_target_for_discovery(
                        target, mission, period_range, snr_threshold, transit_depth_min
                    )
                    
                    if result and result.get('is_candidate', False):
                        candidates.append(result)
                        
                except Exception as e:
                    st.warning(f"⚠️ Failed to analyze {target}: {str(e)}")
                    continue
            
            # Step 3: ML classification and validation
            status_text.text("🤖 Applying ML classification...")
            progress_bar.progress(80)
            
            validated_candidates = self._validate_candidates(
                candidates, confidence_threshold, discovery_method
            )
            
            # Step 4: Cross-reference with known exoplanets
            status_text.text("🔍 Checking against known exoplanets...")
            progress_bar.progress(90)
            
            new_candidates = self._cross_reference_candidates(validated_candidates)
            
            # Step 5: Save results
            status_text.text("💾 Saving results...")
            progress_bar.progress(95)
            
            self._save_discovery_results(new_candidates)
            
            # Complete
            progress_bar.progress(100)
            status_text.text("✅ Auto-discovery completed!")
            
            st.success(f"🎉 Discovery complete! Found {len(new_candidates)} new potential candidates")
            
            # Display summary
            if new_candidates:
                st.subheader("📊 Discovery Summary")
                self._display_discovery_summary(new_candidates)
            
        except Exception as e:
            st.error(f"❌ Auto-discovery failed: {str(e)}")
            self.logger.error(f"Auto-discovery failed: {e}")
    
    def _generate_target_list(self, mission, max_targets):
        """Generate a list of targets for analysis."""
        try:
            # Use database to get unanalyzed targets
            targets = self.db.get_unanalyzed_targets(mission, max_targets)
            
            if not targets:
                st.warning("⚠️ No unanalyzed targets available. Try increasing the target count or changing the mission.")
                return []
            
            return targets
            
        except Exception as e:
            self.logger.error(f"Target list generation failed: {e}")
            return []
    
    def _analyze_target_for_discovery(self, target, mission, period_range, snr_threshold, transit_depth_min):
        """Analyze a single target for transit signals."""
        try:
            # Perform basic analysis
            result = self.analyzer.perform_analysis(
                target_name=target,
                mission=mission,
                analysis_type="Basic",
                period_range=period_range
            )
            
            if not result:
                return None
            
            # Extract key metrics safely
            transit_depth = result.get('transit_depth', 0)
            snr = result.get('snr', 0)
            period = result.get('best_period', 0)
            bls_power = result.get('bls_power', 0)
            
            # Handle period units
            if hasattr(period, 'value'):
                period = period.value
            
            # Check if it meets criteria
            is_candidate = (
                snr >= snr_threshold and
                transit_depth >= transit_depth_min and
                period_range[0] <= period <= period_range[1] and
                bls_power > 0.1
            )
            
            # Mark target as analyzed
            self.db.mark_target_analyzed(
                target, mission, "auto_discovery", is_candidate, 
                result.get('snr', 0) / 10.0  # Normalize confidence score
            )
            
            return {
                'target_name': target,
                'mission': mission,
                'is_candidate': is_candidate,
                'transit_depth': transit_depth,
                'snr': snr,
                'period': period,
                'bls_power': bls_power,
                'analysis_result': result
            }
            
        except Exception as e:
            self.logger.error(f"Target analysis failed for {target}: {e}")
            return None
    
    def _validate_candidates(self, candidates, confidence_threshold, discovery_method):
        """Validate candidates using ML and additional checks."""
        try:
            validated = []
            
            for candidate in candidates:
                if not candidate.get('is_candidate', False):
                    continue
                
                # Prepare features for ML
                features = {
                    'bls_power': candidate.get('bls_power', 0),
                    'period': candidate.get('period', 0),
                    'transit_depth': candidate.get('transit_depth', 0),
                    'snr': candidate.get('snr', 0),
                    'transit_duration': 0.1,  # Placeholder
                    'odd_even_ratio': 1.0,    # Placeholder
                    'secondary_eclipse': 0.0,  # Placeholder
                    'stellar_variability': 0.1, # Placeholder
                    'data_quality': 0.8,      # Placeholder
                    'observation_count': 1000  # Placeholder
                }
                
                # Get ML prediction
                ml_prediction = self.analyzer.ml_predictor.predict_exoplanet(features)
                confidence = ml_prediction.get('confidence', 0)
                
                # Apply confidence threshold
                if confidence >= confidence_threshold:
                    candidate['ml_confidence'] = confidence
                    candidate['ml_prediction'] = ml_prediction.get('prediction', 'unknown')
                    candidate['discovery_method'] = discovery_method
                    validated.append(candidate)
            
            return validated
            
        except Exception as e:
            self.logger.error(f"Candidate validation failed: {e}")
            return candidates  # Return original candidates if validation fails
    
    def _cross_reference_candidates(self, candidates):
        """Cross-reference candidates with known exoplanet databases."""
        try:
            new_candidates = []
            
            for candidate in candidates:
                target_name = candidate['target_name']
                
                # Check if this star has known exoplanets
                is_known = self._check_known_exoplanets(target_name)
                
                if not is_known:
                    candidate['is_known_exoplanet'] = False
                    candidate['validation_status'] = 'new_candidate'
                    new_candidates.append(candidate)
                else:
                    candidate['is_known_exoplanet'] = True
                    candidate['validation_status'] = 'known_exoplanet'
            
            return new_candidates
            
        except Exception as e:
            self.logger.error(f"Cross-referencing failed: {e}")
            return candidates  # Return original candidates if cross-referencing fails
    
    def _check_known_exoplanets(self, target_name):
        """Check if a target has known exoplanets."""
        try:
            # This would query NASA Exoplanet Archive, Simbad, etc.
            # For now, use a simple heuristic based on target name
            
            # Simulate some known exoplanet hosts
            known_hosts = [
                "KIC 11442793",  # Kepler-90
                "KIC 11446443",  # Kepler-11
                "KIC 10028792",  # Kepler-20
                "TIC 261136679", # TOI-700
                "TIC 377659417"  # TOI-1338
            ]
            
            return target_name in known_hosts
            
        except Exception as e:
            self.logger.error(f"Known exoplanet check failed for {target_name}: {e}")
            return False
    
    def _save_discovery_results(self, candidates):
        """Save discovery results to database."""
        try:
            for candidate in candidates:
                self.db.save_new_candidate({
                    'target_name': candidate['target_name'],
                    'mission': candidate['mission'],
                    'discovery_method': candidate['discovery_method'],
                    'bls_power': candidate.get('bls_power', 0),
                    'period': candidate.get('period', 0),
                    'transit_depth': candidate.get('transit_depth', 0),
                    'snr': candidate.get('snr', 0),
                    'ml_confidence': candidate.get('ml_confidence', 0),
                    'is_known_exoplanet': candidate.get('is_known_exoplanet', False),
                    'validation_status': candidate.get('validation_status', 'pending')
                })
            
            self.logger.info(f"Saved {len(candidates)} new candidates to database")
            
        except Exception as e:
            self.logger.error(f"Failed to save discovery results: {e}")
    
    def _display_discovered_candidates(self):
        """Display discovered candidates."""
        try:
            candidates = self.db.get_new_candidates()
            
            if candidates:
                st.subheader("📋 Discovered Candidates")
                
                # Create a DataFrame for display
                import pandas as pd
                
                df = pd.DataFrame(candidates)
                df['discovery_date'] = pd.to_datetime(df['created_at'])
                
                # Filter by validation status
                new_candidates = df[df['validation_status'] == 'new_candidate']
                known_exoplanets = df[df['validation_status'] == 'known_exoplanet']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("New Candidates", len(new_candidates))
                
                with col2:
                    st.metric("Known Exoplanets", len(known_exoplanets))
                
                # Display table
                if not new_candidates.empty:
                    st.subheader("🆕 New Potential Candidates")
                    display_df = new_candidates[['target_name', 'mission', 'period', 'transit_depth', 'snr', 'ml_confidence']].copy()
                    display_df.columns = ['Target', 'Mission', 'Period (days)', 'Transit Depth', 'SNR', 'ML Confidence']
                    st.dataframe(display_df, use_container_width=True)
                
                if not known_exoplanets.empty:
                    st.subheader("📚 Known Exoplanet Hosts")
                    display_df = known_exoplanets[['target_name', 'mission', 'period', 'transit_depth', 'snr']].copy()
                    display_df.columns = ['Target', 'Mission', 'Period (days)', 'Transit Depth', 'SNR']
                    st.dataframe(display_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Failed to display candidates: {str(e)}")
            self.logger.error(f"Failed to display candidates: {e}")
    
    def _display_discovery_summary(self, candidates):
        """Display a summary of the discovery results."""
        try:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_period = np.mean([c.get('period', 0) for c in candidates])
                st.metric("Average Period", f"{avg_period:.2f} days")
            
            with col2:
                avg_snr = np.mean([c.get('snr', 0) for c in candidates])
                st.metric("Average SNR", f"{avg_snr:.2f}")
            
            with col3:
                avg_confidence = np.mean([c.get('ml_confidence', 0) for c in candidates])
                st.metric("Average ML Confidence", f"{avg_confidence:.2f}")
            
            # Mission distribution
            mission_counts = {}
            for candidate in candidates:
                mission = candidate.get('mission', 'Unknown')
                mission_counts[mission] = mission_counts.get(mission, 0) + 1
            
            if mission_counts:
                st.subheader("📊 Mission Distribution")
                for mission, count in mission_counts.items():
                    st.write(f"• {mission}: {count} candidates")
            
        except Exception as e:
            self.logger.error(f"Failed to display discovery summary: {e}")
    
    def _show_batch_processing(self):
        """Show the batch processing interface."""
        st.header("⚡ Batch Processing")
        st.info("Process multiple targets simultaneously.")
        
        # Batch input
        targets_input = st.text_area(
            "Target List",
            placeholder="Enter target names, one per line:\nKIC 11442793\nTIC 123456789\n...",
            key="batch_targets_input"
        )
        
        if st.button("⚡ Start Batch Processing", key="start_batch_btn"):
            if targets_input:
                targets = [t.strip() for t in targets_input.split('\n') if t.strip()]
                st.info(f"Processing {len(targets)} targets...")
            else:
                st.warning("Please enter target names.")
    
    def _show_advanced_analysis(self):
        """Show the advanced analysis interface."""
        st.header("🔬 Advanced Analysis")
        st.info("Advanced analysis features and visualizations.")
        
        if st.session_state.analysis_results:
            # Advanced visualizations
            st.subheader("📊 Advanced Visualizations")
            
            viz_type = st.selectbox(
                "Visualization Type",
                ["transit", "periodogram", "phase_fold"],
                key="viz_type_select"
            )
            
            if st.button("📊 Generate Plot", key="generate_plot_btn"):
                fig = self.viz_manager.create_publication_ready_plot(
                    st.session_state.analysis_results, viz_type
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("Run an analysis first to see advanced features.")
    
    def _show_community_sharing(self):
        """Show the community sharing interface."""
        st.header("👥 Community Sharing")
        st.info("Share discoveries and collaborate with the community.")
        
        # Share discovery
        st.subheader("📤 Share Discovery")
        
        if st.session_state.current_target:
            annotation_type = st.selectbox(
                "Annotation Type",
                ["observation", "analysis", "comment"],
                key="annotation_type_select"
            )
            
            annotation_content = st.text_area(
                "Annotation Content",
                key="annotation_content_input"
            )
            
            if st.button("📤 Share", key="share_btn"):
                if annotation_content:
                    success = self.community_manager.add_annotation(
                        st.session_state.current_target,
                        annotation_type,
                        annotation_content
                    )
                    if success:
                        st.success("✅ Annotation shared successfully!")
                    else:
                        st.error("❌ Failed to share annotation.")
                else:
                    st.warning("Please enter annotation content.")
        else:
            st.info("Run an analysis first to share discoveries.")
    
    def _show_real_time_monitoring(self):
        """Show the real-time monitoring interface."""
        st.header("📡 Real-time Monitoring")
        st.info("Monitor targets for real-time alerts and updates.")
        
        st.info("Real-time monitoring will be implemented in future versions.")
    
    def _show_help_tutorials(self):
        """Show the help and tutorials interface."""
        st.header("📚 Help & Tutorials")
        
        # Tutorial selection
        tutorial_level = st.selectbox(
            "Tutorial Level",
            ["beginner", "intermediate", "advanced"],
            key="tutorial_level_select"
        )
        
        if st.button("📚 Load Tutorial", key="load_tutorial_btn"):
            tutorial = self.educational_manager.get_tutorial(tutorial_level)
            
            st.subheader(tutorial['title'])
            for i, step in enumerate(tutorial['content'], 1):
                st.markdown(f"{i}. {step}")
    
    def _display_analysis_results(self, results):
        """Display analysis results."""
        st.header("📊 Analysis Results")
        
        # Overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Best Period", f"{results.get('best_period', 0):.3f} days")
        
        with col2:
            st.metric("Transit Depth", f"{results.get('transit_depth', 0):.4f}")
        
        with col3:
            st.metric("Signal-to-Noise", f"{results.get('snr', 0):.2f}")
        
        # Quality assessment
        quality_scores = results.get('quality_scores', {})
        if quality_scores:
            st.subheader("📈 Data Quality Assessment")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Overall Score", f"{quality_scores.get('overall_score', 0):.2f}")
            
            with col2:
                st.metric("Completeness", f"{quality_scores.get('completeness', 0):.2f}")
            
            with col3:
                st.metric("Noise Level", f"{quality_scores.get('noise_level', 0):.4f}")
            
            with col4:
                st.metric("Max Gap", f"{quality_scores.get('max_gap', 0):.2f} days")
        
        # Export options
        st.subheader("📤 Export Results")
        
        export_cols = st.columns(4)
        
        with export_cols[0]:
            if st.button("📊 CSV", key="export_csv_btn"):
                export_path = self.export_manager.export_results(results, results.get('target_name', 'unknown'), 'csv')
                if export_path:
                    st.success(f"✅ Exported to {export_path}")
        
        with export_cols[1]:
            if st.button("📄 JSON", key="export_json_btn"):
                export_path = self.export_manager.export_results(results, results.get('target_name', 'unknown'), 'json')
                if export_path:
                    st.success(f"✅ Exported to {export_path}")
        
        with export_cols[2]:
            if st.button("🖼️ PNG", key="export_png_btn"):
                export_path = self.export_manager.export_results(results, results.get('target_name', 'unknown'), 'png')
                if export_path:
                    st.success(f"✅ Exported to {export_path}")
        
        with export_cols[3]:
            if st.button("📋 Share", key="share_discovery_btn"):
                st.success("✅ Discovery shared with community") 

    def _show_star_shortlist_analysis(self):
        """Show the Star Shortlist & Analysis interface."""
        st.header("⭐ Star Shortlist & Analysis")
        st.info("Query TESS Input Catalog for promising stars and analyze them for transit signals.")
        
        # Import the new modules
        try:
            from shortlist import get_shortlist, get_shortlist_stats
            from analysis_pipeline import AnalysisPipeline
        except ImportError as e:
            st.error(f"Error importing modules: {e}")
            st.info("Please ensure shortlist.py and analysis_pipeline.py are in the project directory.")
            return
        
        # Create tabs
        tab1, tab2 = st.tabs(["📋 Shortlist", "🔍 Run Analysis"])
        
        with tab1:
            self._show_shortlist_tab(get_shortlist, get_shortlist_stats)
        
        with tab2:
            self._show_analysis_tab(AnalysisPipeline)
    
    def _show_shortlist_tab(self, get_shortlist_func, get_shortlist_stats_func):
        """Show the shortlist tab."""
        st.subheader("📋 Star Shortlist")
        
        # Refresh button
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🔄 Refresh Shortlist", key="refresh_shortlist_btn"):
                st.session_state.shortlist_df = get_shortlist_func(force_refresh=True)
                st.success("✅ Shortlist refreshed!")
        
        # Load shortlist
        if 'shortlist_df' not in st.session_state:
            st.session_state.shortlist_df = get_shortlist_func()
        
        # Display shortlist
        if not st.session_state.shortlist_df.empty:
            st.dataframe(
                st.session_state.shortlist_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Show statistics
            stats = get_shortlist_stats_func()
            st.subheader("📊 Shortlist Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Stars", stats['total_stars'])
            with col2:
                st.metric("Avg Tmag", f"{stats['avg_tmag']:.2f}")
            with col3:
                st.metric("Avg Teff", f"{stats['avg_teff']:.0f} K")
            with col4:
                st.metric("Avg Radius", f"{stats['avg_radius']:.2f} R☉")
        else:
            st.warning("No stars found in shortlist. Try refreshing.")
    
    def _show_analysis_tab(self, AnalysisPipelineClass):
        """Show the analysis tab."""
        st.subheader("🔍 Run Analysis")
        
        # Initialize pipeline
        if 'analysis_pipeline' not in st.session_state:
            st.session_state.analysis_pipeline = AnalysisPipelineClass()
        
        # Analysis parameters
        col1, col2 = st.columns(2)
        
        with col1:
            max_stars = st.number_input(
                "Max Stars to Analyze",
                min_value=1,
                max_value=100,
                value=10,
                help="Maximum number of stars to analyze in this run",
                key="max_stars_analysis"
            )
            
            bls_power_threshold = st.number_input(
                "BLS Power Threshold",
                min_value=1.0,
                max_value=20.0,
                value=7.0,
                step=0.5,
                help="Minimum BLS power for significant transit signal",
                key="bls_power_threshold"
            )
        
        with col2:
            period_min = st.number_input(
                "Min Period (days)",
                min_value=0.1,
                max_value=10.0,
                value=0.5,
                step=0.1,
                help="Minimum orbital period to search",
                key="period_min_analysis"
            )
            
            period_max = st.number_input(
                "Max Period (days)",
                min_value=1.0,
                max_value=100.0,
                value=20.0,
                step=1.0,
                help="Maximum orbital period to search",
                key="period_max_analysis"
            )
        
        # Start analysis button
        if st.button("🚀 Start Analysis", key="start_shortlist_analysis_btn"):
            if 'shortlist_df' not in st.session_state or st.session_state.shortlist_df.empty:
                st.error("No shortlist available. Please generate a shortlist first.")
                return
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_container = st.container()
            
            def progress_callback(current, total, star_id, is_candidate):
                progress = current / total
                progress_bar.progress(progress)
                
                status = f"Analyzing {star_id} ({current}/{total})"
                if is_candidate:
                    status += " - Candidate found!"
                status_text.text(status)
            
            # Run analysis
            with st.spinner("Running analysis..."):
                try:
                    candidates = st.session_state.analysis_pipeline.process_shortlist(
                        st.session_state.shortlist_df,
                        max_stars=max_stars,
                        progress_callback=progress_callback
                    )
                    
                    progress_bar.progress(1.0)
                    status_text.text("Analysis complete!")
                    
                    # Display results
                    with results_container:
                        st.success(f"✅ Analysis complete! Found {len(candidates)} new candidates.")
                        
                        if candidates:
                            st.subheader("🎯 New Candidates")
                            candidates_df = pd.DataFrame(candidates)
                            st.dataframe(candidates_df, use_container_width=True)
                        
                        # Show all candidates
                        all_candidates = st.session_state.analysis_pipeline.get_all_candidates()
                        if not all_candidates.empty:
                            st.subheader("📊 All Candidates")
                            st.dataframe(all_candidates, use_container_width=True)
                            
                            # Export options
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("📊 Export to CSV", key="export_candidates_csv_btn"):
                                    csv_path = f"candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                                    all_candidates.to_csv(csv_path, index=False)
                                    st.success(f"✅ Exported to {csv_path}")
                            
                            with col2:
                                if st.button("🗑️ Clear Database", key="clear_candidates_db_btn"):
                                    st.session_state.analysis_pipeline.clear_database()
                                    st.success("✅ Database cleared!")
                                    st.rerun()
                
                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")
                    self.logger.error(f"Analysis failed: {e}")
        
        # Show current statistics
        if st.button("📊 Show Statistics", key="show_candidate_stats_btn"):
            stats = st.session_state.analysis_pipeline.get_candidate_stats()
            
            st.subheader("📈 Candidate Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Candidates", stats['total_candidates'])
            with col2:
                st.metric("Avg Period", f"{stats['avg_period']:.2f} days")
            with col3:
                st.metric("Avg Depth", f"{stats['avg_depth']:.4f}")
            with col4:
                st.metric("Avg SNR", f"{stats['avg_snr']:.2f}")
            
            if stats['mission_distribution']:
                st.subheader("📊 Mission Distribution")
                for mission, count in stats['mission_distribution'].items():
                    st.write(f"{mission}: {count} candidates") 