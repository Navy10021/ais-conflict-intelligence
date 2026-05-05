# Maritime Conflict Intelligence System (MCIS)

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comprehensive system for detecting maritime behavioral anomalies correlated with armed conflicts using AIS (Automatic Identification System) data from the Danish Maritime Authority.

## 🎯 Mission

Process, analyze, and model AIS data to detect maritime behavioral anomalies correlated with armed conflicts and build a predictive early-warning system at publication quality.

**Core Research Question:**
> "Do maritime behavioral signals — traffic density, vessel-type composition, speed deviation, AIS silence, rate of turn, and route entropy — exhibit statistically significant changes before and after armed conflict onset? Can a temporal model exploit these leading indicators to predict conflict events in advance?"

## 🌍 Target Conflict Zones

| Conflict | Onset | Key Zone |
|----------|-------|----------|
| Russia–Ukraine War | 2022-02-24 | Black Sea, Sea of Azov, Kerch Strait |
| Red Sea / Houthi Crisis | 2023-11-19 | Red Sea, Gulf of Aden, Bab-el-Mandeb |
| Taiwan Strait Tensions | 2022-08-04 | Taiwan Strait |
| South China Sea Disputes | Ongoing | SCS, Paracel & Spratly Islands |
| Iran–Gulf Tensions | Ongoing | Strait of Hormuz |

## 📁 Project Structure

```
ais-conflict-intelligence/
├── config/                    # Configuration files
│   └── settings.yaml
├── data/                      # Raw AIS data (not committed)
│   └── YYYY-MM-DD/
│       └── aisdk-YYYY-MM-DD.csv
├── data_external/             # External datasets (not committed)
├── data_sample/               # Sample data for testing
├── notebooks/                 # Jupyter notebooks for EDA and analysis
│   ├── 01_EDA.ipynb
│   ├── 02_preprocessing_validation.ipynb
│   ├── 03_visualization.ipynb
│   ├── 04_conflict_correlation.ipynb
│   └── 05_model_development.ipynb
├── outputs/                   # Generated outputs (not committed)
├── scripts/                   # Pipeline scripts
│   ├── run_pipeline.sh
│   └── run_pipeline.py
├── src/                       # Source code
│   ├── preprocessing/         # Data loading, cleaning, feature engineering
│   ├── visualization/         # Spatial, temporal, and statistical viz
│   ├── analysis/              # Correlation and behavioral analysis
│   └── models/               # Anomaly detection and conflict prediction
├── tests/                     # Unit tests
├── requirements.txt           # Python dependencies
├── pyproject.toml            # Package configuration
├── README.md
├── LICENSE
└── .gitignore
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/Navy10021/ais-conflict-intelligence.git
cd ais-conflict-intelligence

# Create virtual environment (recommended)
conda create -n mcis python=3.11 -y
conda activate mcis

# Install dependencies
pip install -e .
```

### Data Setup

1. Download AIS data from [Danish Maritime Authority](https://www.dma.dk/)
2. Place data in `data/YYYY-MM-DD/aisdk-YYYY-MM-DD.csv` format
3. (Optional) Add external datasets to `data_external/`

### Running the Pipeline

```bash
# Full pipeline (Linux/Mac)
bash scripts/run_pipeline.sh

# Full pipeline (Cross-platform Python)
python scripts/run_pipeline.py

# Individual steps
python -m src.preprocessing.loader --data-root ./data --output outputs/processed/ais_raw.parquet --verbose
python -m src.preprocessing.cleaner --input outputs/processed/ais_raw.parquet --output outputs/processed/ais_clean.parquet --verbose
python -m src.preprocessing.feature_engineer --input outputs/processed/ais_clean.parquet --output outputs/processed/ais_features.parquet --verbose
```

## 📊 Features

### Preprocessing Pipeline
- **Loader**: Multi-day AIS data loading with date filtering
- **Cleaner**: Data validation, coordinate filtering, MMSI validation, missing value imputation
- **Feature Engineer**: 6 categories of features (Kinematic, ROT, Geospatial, Behavioral, Destination/ETA, Temporal Aggregation)
- **Aggregator**: Grid-cell × time-bucket aggregation

### Analysis Modules
- **Traffic Analyzer**: Volume and density analysis
- **Behavioral Analyzer**: Vessel behavior pattern analysis
- **Correlation Analyzer**: Granger causality, DiD, ITS, Event Study
- **ROT Analyzer**: Rate-of-Turn anomaly analysis
- **Destination Analyzer**: Port call and destination analysis

### Models
- **Anomaly Detection**: Isolation Forest, VAE, DBSCAN, LOF
- **Conflict Predictor**: LSTM-Attention, TFT, XGBoost, ARIMA/Prophet baselines

## 📈 Key Features Generated

| Category | Features |
|----------|-----------|
| Kinematic | speed_category, delta_sog, delta_cog, is_dark_ship, sog_z_score |
| ROT | rot_category, rot_abs, rot_spike, evasive_maneuver |
| Geospatial | grid_cell, in_conflict_zone, dist_chokepoint_km |
| Behavioral | route_entropy, loitering_flag, zig_zag_index, rolling statistics |
| Destination | dest_is_conflict_port, eta_hours_remaining, dest_changed |
| Temporal Agg | traffic_count, military_ratio, dark_ship_ratio, mean_rot_abs |

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_loader.py -v
```

## 📖 Documentation

Detailed documentation is available in the `notebooks/` directory:
- `01_EDA.ipynb`: Exploratory Data Analysis
- `02_preprocessing_validation.ipynb`: Preprocessing validation
- `03_visualization.ipynb`: Visualization examples
- `04_conflict_correlation.ipynb`: Correlation analysis
- `05_model_development.ipynb`: Model development and evaluation

## 📄 Paper Structure

```
Title: "Maritime Traffic Anomaly Detection as a Precursor to Armed Conflict:
        Evidence from AIS Data in Global Hotspots (2022–2026)"
```

See `CLAUDE.md` for detailed paper structure and methodology.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@misc{mcis2026,
  author = {Yoonseop Lee},
  title = {Maritime Conflict Intelligence System: AIS-Based Conflict Early Warning},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Navy10021/ais-conflict-intelligence}}
}
```

## 📧 Contact

Your Name - iyunseob4@gmail.com

Project Link: [https://github.com/Navy10021/ais-conflict-intelligence](https://github.com/Navy10021/ais-conflict-intelligence)

## 🙏 Acknowledgments

- Danish Maritime Authority for providing AIS data
- ACLED for conflict event data
- World Port Index for port location data
