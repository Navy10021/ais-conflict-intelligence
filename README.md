# Maritime Conflict Intelligence System (MCIS)

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Alpha](https://img.shields.io/badge/Status-Alpha-orange)](pyproject.toml)

> **AIS-driven maritime anomaly detection and conflict early-warning research framework**

MCIS is a research and engineering project that extracts vessel behavior changes from large-scale AIS time-series data and analyzes their correlation with armed conflict onset to test whether maritime traffic changes can serve as conflict precursors.

---

## ✨ Why MCIS?

- **End-to-end pipeline**: ingest → clean → feature engineer → aggregate → analyze → visualize → model.
- **Conflict-aware features**: behavior-based signals such as AIS silence, ROT spikes, route entropy, and military/civilian composition.
- **Reproducible research structure**: configuration-driven (`config/settings.yaml`) with modular CLIs and tests.
- **Paper-ready design**: combines Event Study, ITS, Granger, and DiD analysis with predictive modeling.

---

## 🎯 Core Research Question

> “Do maritime behavioral signals—traffic density, vessel-type composition, speed/course variability, AIS silence, ROT anomalies, and route entropy—change significantly before and after conflict onset, and can these leading indicators be used to predict conflict events in advance?”

---

## 🌍 Target Conflict Zones

| Conflict | Onset | Key Zone |
|---|---:|---|
| Russia–Ukraine War | 2022-02-24 | Black Sea, Sea of Azov, Kerch Strait |
| Red Sea / Houthi Crisis | 2023-11-19 | Red Sea, Gulf of Aden, Bab-el-Mandeb |
| Taiwan Strait Tensions | 2022-08-04 | Taiwan Strait |
| South China Sea Disputes | Ongoing | SCS, Paracel & Spratly Islands |
| Iran–Gulf Tensions | Ongoing | Strait of Hormuz |

---

## 🧱 System Architecture

```text
Raw AIS CSV (daily)
    ↓
[Loader] Date-range ingestion
    ↓
[Cleaner] Quality and physical-rule cleaning
    ↓
[Feature Engineer] Behavioral, geospatial, and destination features
    ↓
[Aggregator] Grid × time-bucket aggregation
    ↓
┌───────────────┬───────────────────┬────────────────────┐
│ Visualization │ Statistical Analysis│ Predictive Modeling │
└───────────────┴───────────────────┴────────────────────┘
```

---

## 📦 Project Layout

```bash
ais-conflict-intelligence/
├── config/settings.yaml
├── data/                     # Raw AIS data (daily folders)
├── data_external/            # External data (e.g., ACLED, world_ports)
├── docs/report.md
├── notebooks/                # EDA, validation, visualization, correlation, modeling
├── scripts/
│   ├── run_pipeline.py
│   ├── run_pipeline.sh
│   └── generate_report.py
├── src/
│   ├── preprocessing/        # loader, cleaner, feature_engineer, aggregator
│   ├── visualization/        # spatial/temporal/statistical/conflict overlay
│   ├── analysis/             # traffic/behavioral/correlation/ROT/destination
│   └── models/               # anomaly, predictor, baseline, evaluator
└── tests/                    # unit tests
```

---

## 🚀 Quick Start

### 1) Installation

```bash
git clone https://github.com/Navy10021/ais-conflict-intelligence.git
cd ais-conflict-intelligence

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -e .
```

> Optional extras example: `pip install -e .[dev,viz,ml,forecasting,notebooks]`

### 2) Data Setup

1. Obtain AIS data from the Danish Maritime Authority.
2. Store files in the following format:
   - `data/YYYY-MM-DD/aisdk-YYYY-MM-DD.csv`
3. Place external data (ACLED, ports) under `data_external/`:
   - `data_external/acled_events.csv`
   - `data_external/world_ports.csv`

### 3) Run Full Pipeline

```bash
# Bash wrapper
bash scripts/run_pipeline.sh

# Cross-platform Python CLI
python scripts/run_pipeline.py --date-start 2024-03-01 --date-end 2024-08-31 --verbose
```

### 4) Run by Stage (CLI Modules)

```bash
python -m src.preprocessing.loader --data-root ./data --output outputs/processed/ais_raw.parquet --verbose
python -m src.preprocessing.cleaner --input outputs/processed/ais_raw.parquet --output outputs/processed/ais_clean.parquet --verbose
python -m src.preprocessing.feature_engineer --input outputs/processed/ais_clean.parquet --output outputs/processed/ais_features.parquet --conflict-events data_external/acled_events.csv --ports data_external/world_ports.csv --verbose
python -m src.preprocessing.aggregator --input outputs/processed/ais_features.parquet --output outputs/processed/ais_aggregated.parquet --verbose
```

---

## 🧠 Feature Space (Examples)

| Category | Representative Features |
|---|---|
| Kinematic | `delta_sog`, `delta_cog`, `speed_category`, `sog_z_score` |
| ROT / Maneuver | `rot_abs`, `rot_spike`, `evasive_maneuver` |
| Geospatial | `grid_cell`, `in_conflict_zone`, `dist_chokepoint_km` |
| Behavioral | `route_entropy`, `loitering_flag`, `zig_zag_index` |
| Destination/ETA | `dest_changed`, `dest_is_conflict_port`, `eta_hours_remaining` |
| Temporal Aggregation | `traffic_count`, `military_ratio`, `dark_ship_ratio` |

---

## 📊 Analysis & Modeling Modules

### Analysis
- `traffic_analyzer`: traffic volume, density, and composition shifts
- `behavioral_analyzer`: behavior patterns and anomaly signals
- `correlation_analyzer`: correlation/causal exploration with Granger, DiD, ITS, and Event Study
- `rot_analyzer`: sharp-turn and evasive-maneuver anomaly signals
- `destination_analyzer`: route changes using destination, port-call, and ETA patterns

### Models
- `anomaly_model`: anomaly detection with Isolation Forest, DBSCAN, and LOF
- `conflict_predictor`: time-series conflict prediction training pipeline
- `baseline`: baseline comparison models
- `evaluator`: evaluation metrics and reporting

---

## 🧪 Testing

```bash
pytest tests/
pytest tests/test_loader.py -v
```

---

## 📚 Documentation

- `notebooks/01_EDA.ipynb` — exploratory data analysis
- `notebooks/02_preprocessing_validation.ipynb` — preprocessing validation
- `notebooks/03_visualization.ipynb` — visualization workflows
- `notebooks/04_conflict_correlation.ipynb` — correlation and event analysis
- `notebooks/05_model_development.ipynb` — modeling and evaluation
- `docs/report.md` — draft project report

---

## 🛠️ Configuration Highlights

Key execution parameters are managed in `config/settings.yaml`.

- data and output paths
- analysis window (`date_range`)
- conflict-zone bounding boxes and chokepoint coordinates
- preprocessing granularity (`grid_size`, `time_bucket`)
- model training hyperparameters
- cleaning and validation thresholds

---

## 🗺️ Output Convention

Default output root is `./outputs`.

- `outputs/processed/`: intermediate parquet artifacts
- `outputs/figures/`: generated visualizations
- `outputs/tables/`: statistical tables
- `outputs/models/`: model artifacts
- `outputs/reports/`: report files

---

## 🤝 Contributing

Pull requests are welcome.

1. Fork
2. Create a feature branch
3. Make changes and test
4. Commit and push
5. Open a pull request

---

## 📜 License

Licensed under MIT. See [LICENSE](LICENSE) for details.

## 📧 Contact

- Author: Yoonseop Lee
- Email: iyunseob4@gmail.com
- Project: https://github.com/Navy10021/ais-conflict-intelligence
