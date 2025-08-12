# Advanced Exoplanet Detector

A professional, modular exoplanet detection tool with advanced analysis capabilities, built with clean architecture and comprehensive features.

## Features

### Core Detection
- **Light Curve Analysis**: Advanced transit detection using BLS and Lomb-Scargle periodograms
- **Multi-mission Support**: Kepler, TESS, and K2 data processing
- **Quality Assessment**: Comprehensive data quality evaluation
- **False Positive Analysis**: Robust candidate validation
- **Star Shortlist & Analysis**: Automated TESS catalog querying and batch analysis pipeline

### Advanced Analysis
- **Transit Modeling**: Analytical and numerical transit fitting
- **Stellar Characterization**: Stellar property estimation
- **Machine Learning**: ML-based candidate classification
- **Performance Monitoring**: Real-time performance tracking

### Professional Features
- **Modular Architecture**: Clean, maintainable code structure
- **Database Integration**: SQLite persistence for results and history
- **Caching System**: Intelligent result caching for efficiency
- **Export Capabilities**: Multiple format export (CSV, JSON, PNG, PDF)
- **Community Features**: Annotation and sharing system
- **Educational Tools**: Interactive tutorials and learning materials

## Architecture

The application follows a professional modular architecture:

```
Exoplanet-detector/
├── app.py                    # Main application entry point
├── core/                     # Core functionality modules
│   ├── database.py          # Database management
│   ├── processor.py         # Resilient data processing
│   ├── monitor.py           # Performance monitoring
│   ├── cache.py             # Caching system
│   ├── parallel.py          # Parallel processing
│   ├── ml_manager.py        # Machine learning management
│   ├── real_time.py         # Real-time monitoring
│   ├── community.py         # Community features
│   ├── educational.py       # Educational content
│   ├── quality.py           # Data quality assessment
│   ├── visualization.py     # Plot generation
│   └── export.py            # Export functionality
├── analysis/                 # Analysis modules
│   ├── analyzer.py          # Main analysis engine
│   ├── transit_modeling.py  # Transit modeling
│   ├── false_positive.py    # False positive analysis
│   ├── stellar_characterization.py  # Stellar analysis
│   └── ml_predictor.py      # ML prediction
├── ui/                      # User interface modules
│   ├── interface.py         # Main Streamlit interface
│   └── displays.py          # Result display management
├── shortlist.py             # Star shortlist generation from TESS catalog
├── analysis_pipeline.py     # Batch analysis and persistence pipeline
└── requirements.txt          # Python dependencies
```

## Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Exoplanet-detector
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

### Basic Usage

1. Open the application in your web browser
2. Enter a target name (e.g., "KIC 11442793")
3. Select mission and analysis parameters
4. Click "Start Analysis" to begin detection
5. Review results and export as needed

## Star Shortlist & Analysis

The application includes an automated pipeline for discovering potential exoplanet host stars:

### Star Shortlist Module
- **TESS Catalog Querying**: Automatically queries the TESS Input Catalog for stars matching exoplanet host criteria
- **Selection Criteria**: Tmag < 12, Radius < 1.5 R☉, 3000 K < Teff < 6500 K, CDPP4_0 < 100 ppm
- **CSV Persistence**: Saves results to `star_shortlist.csv` for reuse

### Analysis Pipeline
- **SQLite Database**: Persistent storage in `candidates.db` with `candidates` table
- **BLS Analysis**: Runs Box Least Squares with configurable parameters (0.5-20 days, 10,000 steps)
- **Signal Detection**: Identifies candidates with BLS power ≥ 7.0
- **Progress Tracking**: Real-time progress updates during batch analysis
- **Duplicate Prevention**: Skips already analyzed stars to avoid redundant work

### Usage
1. Navigate to "Star Shortlist & Analysis" in the sidebar
2. **Shortlist Tab**: Click "Refresh Shortlist" to query TESS catalog
3. **Run Analysis Tab**: Configure parameters and click "Start Analysis"
4. Monitor progress and review results in real-time
5. Export candidates or clear database as needed

## Analysis Types

### Basic Analysis
- Light curve preprocessing
- BLS periodogram analysis
- Basic transit detection
- Quality assessment

### Advanced Analysis
- False positive analysis
- Transit modeling
- Stellar characterization
- Confidence scoring

### Comprehensive Analysis
- All advanced features
- Machine learning prediction
- Multi-mission data fusion
- Detailed reporting

## Configuration

### Analysis Parameters
- **Period Range**: 0.1 - 100 days
- **Data Quality**: good/medium/poor
- **Detrending**: flatten/spline/polynomial
- **Mission**: Kepler/TESS/K2

### Database Configuration
- **SQLite Database**: `exoplanet_detector.db`
- **Cache Directory**: `cache/`
- **Export Directory**: `exports/`
- **Models Directory**: `models/`

## Performance Features

### Caching System
- Intelligent result caching
- Configurable cache expiration
- Automatic cache cleanup

### Performance Monitoring
- Operation timing tracking
- Error rate monitoring
- Success rate metrics

### Parallel Processing
- Batch processing capabilities
- Multi-threaded operations
- Configurable worker pools

## Development

### Code Structure
The application follows clean architecture principles:
- **Separation of Concerns**: Each module has a specific responsibility
- **Dependency Injection**: Components are loosely coupled
- **Error Handling**: Comprehensive error handling and logging
- **Testing**: Modular design enables easy testing

### Adding New Features
1. Create new module in appropriate directory
2. Implement required interfaces
3. Add to main application initialization
4. Update documentation

### Contributing
1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

## Scientific Background

### Transit Detection
- **BLS Algorithm**: Box Least Squares for periodic transit detection
- **Lomb-Scargle**: Alternative periodogram method
- **Signal Processing**: Advanced filtering and detrending

### False Positive Analysis
- **SNR Analysis**: Signal-to-noise ratio evaluation
- **Transit Duration**: Duration vs period analysis
- **Odd-Even Analysis**: Transit consistency checking
- **Secondary Eclipse**: Secondary eclipse detection

### Data Quality Assessment
- **Completeness**: Data coverage analysis
- **Noise Level**: Photometric noise evaluation
- **Gap Analysis**: Data gap identification
- **Systematic Errors**: Instrumental effect detection

## Support

### Documentation
- Comprehensive inline documentation
- API reference for all modules
- Usage examples and tutorials

### Community
- GitHub Issues for bug reports
- Discussion forum for questions
- Contributing guidelines

### Acknowledgments
- NASA Exoplanet Archive for data access
- LightKurve team for astronomical data processing
- Streamlit team for the web framework
- Scientific community for algorithms and methods

## Recent Updates

### Enhanced Star Shortlist Module
- Added advanced star filtering functions
- Implemented data quality assessment
- Enhanced statistics with comprehensive metrics
- Improved error handling and logging

### Advanced Processing Pipeline
- Added batch processing capabilities with parallel execution
- Implemented performance monitoring and metrics tracking
- Enhanced error handling with detailed performance reporting
- Added timeout and retry mechanisms for robust processing

### Performance Improvements
- Thread-safe performance metrics collection
- Memory-efficient batch processing
- Optimized data structures and algorithms
- Enhanced caching and resource management

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built for the exoplanet discovery community**

