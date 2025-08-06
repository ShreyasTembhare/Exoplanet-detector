# Advanced Exoplanet Detector

A professional, modular exoplanet detection tool with advanced analysis capabilities, built with a clean architecture and comprehensive feature set.

## 🌟 Features

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

## 🏗️ Architecture

The application follows a professional modular architecture:

```
Exoplanet-detector/
├── app.py                    # Main application entry point
├── core/                     # Core functionality modules
│   ├── __init__.py
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
│   ├── __init__.py
│   ├── analyzer.py          # Main analysis engine
│   ├── transit_modeling.py  # Transit modeling
│   ├── false_positive.py    # False positive analysis
│   ├── stellar_characterization.py  # Stellar analysis
│   └── ml_predictor.py      # ML prediction
├── ui/                      # User interface modules
│   ├── __init__.py
│   ├── interface.py         # Main Streamlit interface
│   └── displays.py          # Result display management
├── shortlist.py             # Star shortlist generation from TESS catalog
├── analysis_pipeline.py     # Batch analysis and persistence pipeline
├── requirements.txt          # Python dependencies
├── README.md               # This documentation
└── LICENSE                 # MIT License
```

## 🚀 Quick Start

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

1. **Open the application** in your web browser
2. **Enter a target name** (e.g., "KIC 11442793")
3. **Select mission and analysis parameters**
4. **Click "Start Analysis"** to begin detection
5. **Review results** and export as needed

## 📊 Analysis Types

### Basic Analysis
- Light curve preprocessing
- BLS periodogram analysis
- Basic transit detection
- Quality assessment

## ⭐ Star Shortlist & Analysis

The application now includes an automated pipeline for discovering potential exoplanet host stars:

### Star Shortlist Module (`shortlist.py`)
- **TESS Catalog Querying**: Automatically queries the TESS Input Catalog for stars matching exoplanet host criteria
- **Selection Criteria**: 
  - Tmag < 12 (bright enough for good photometry)
  - Radius < 1.5 R☉ (smaller stars = deeper transits)
  - 3000 K < Teff < 6500 K (main sequence stars)
  - CDPP4_0 < 100 ppm (low noise)
- **CSV Persistence**: Saves results to `star_shortlist.csv` for reuse
- **Statistics**: Provides comprehensive statistics about the shortlist

### Analysis Pipeline (`analysis_pipeline.py`)
- **SQLite Database**: Persistent storage in `candidates.db` with `candidates` table
- **BLS Analysis**: Runs Box Least Squares with configurable parameters (0.5-20 days, 10,000 steps)
- **Signal Detection**: Identifies candidates with BLS power ≥ 7.0
- **Progress Tracking**: Real-time progress updates during batch analysis
- **Duplicate Prevention**: Skips already analyzed stars to avoid redundant work
- **Export Capabilities**: CSV export and database management

### Streamlit Integration
- **Two-Tab Interface**: 
  - **Shortlist Tab**: Display and refresh star shortlist with statistics
  - **Run Analysis Tab**: Configure and execute batch analysis with progress tracking
- **Real-time Updates**: Live progress bars and status updates
- **Results Display**: Comprehensive candidate tables with filtering and export options
- **Database Management**: Clear database and view all candidates functionality

### Usage
1. Navigate to "Star Shortlist & Analysis" in the sidebar
2. **Shortlist Tab**: Click "Refresh Shortlist" to query TESS catalog
3. **Run Analysis Tab**: Configure parameters and click "Start Analysis"
4. Monitor progress and review results in real-time
5. Export candidates or clear database as needed

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

## 🔧 Configuration

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

## 📈 Performance Features

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

## 🎓 Educational Features

### Interactive Tutorials
- Beginner: Introduction to exoplanet detection
- Intermediate: Advanced transit analysis
- Advanced: Machine learning applications

### Learning Materials
- Step-by-step guides
- Interactive examples
- Progress tracking

## 👥 Community Features

### Annotation System
- Add observations and comments
- Rate and review annotations
- Community collaboration

### Discovery Sharing
- Share candidate discoveries
- Export results in multiple formats
- Community validation

## 🔍 Advanced Features

### Real-time Monitoring
- Target monitoring capabilities
- Alert system for significant events
- Continuous data analysis

### Machine Learning
- Pre-trained models for candidate classification
- Feature engineering for transit detection
- Model performance tracking

### Export Capabilities
- **CSV**: Tabular data export
- **JSON**: Structured data export
- **PNG**: High-quality plot export
- **PDF**: Publication-ready reports

## 🛠️ Development

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

## 📚 Scientific Background

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

## 🤝 Support

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

## 🔧 Recent Updates

### Enhanced Star Shortlist Module
- Added `filter_stars_by_criteria()` function for advanced star filtering
- Implemented `validate_star_data()` for data quality assessment
- Enhanced statistics with min/max values for better analysis
- Improved error handling and logging throughout

### Advanced Processing Pipeline
- Added batch processing capabilities with parallel execution
- Implemented performance monitoring and metrics tracking
- Enhanced error handling with detailed performance reporting
- Added timeout and retry mechanisms for robust processing

### Centralized Configuration System
- Created `config.py` for centralized project configuration
- Added comprehensive parameter management for all modules
- Implemented configuration validation and error checking
- Support for environment variable overrides

### Performance Improvements
- Thread-safe performance metrics collection
- Memory-efficient batch processing
- Optimized data structures and algorithms
- Enhanced caching and resource management

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️ for the exoplanet discovery community**

