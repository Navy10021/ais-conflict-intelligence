# CLAUDE.md — Maritime Conflict Intelligence System (MCIS)
# Data source: Danish Maritime Authority AIS (aisdk)

---

## PROJECT OVERVIEW

**Mission**: Process, analyze, and model AIS data from the Danish Maritime Authority
(`aisdk-YYYY-MM-DD.csv`) to detect maritime behavioral anomalies correlated with
armed conflicts and build a predictive early-warning system at publication quality.

**Core Research Question**:
> "Do maritime behavioral signals — traffic density, vessel-type composition, speed
> deviation, AIS silence, rate of turn, and route entropy — exhibit statistically
> significant changes before and after armed conflict onset? Can a temporal model
> exploit these leading indicators to predict conflict events in advance?"

**Target Conflict Zones**:
| Conflict | Onset | Key Zone |
|----------|-------|----------|
| Russia–Ukraine War | 2022-02-24 | Black Sea, Sea of Azov, Kerch Strait |
| Red Sea / Houthi Crisis | 2023-11-19 | Red Sea, Gulf of Aden, Bab-el-Mandeb |
| Taiwan Strait Tensions | 2022-08-04 (PLA drill) | Taiwan Strait |
| South China Sea Disputes | Ongoing | SCS, Paracel & Spratly Islands |
| Iran–Gulf Tensions | Ongoing | Strait of Hormuz |

---

## DATA SOURCE & DIRECTORY LAYOUT

### Raw Data Structure
```
./data/
├── 2024-03-01/
│   └── aisdk-2024-03-01.csv
├── 2024-03-02/
│   └── aisdk-2024-03-02.csv
├── 2024-03-03/
│   └── aisdk-2024-03-03.csv
└── ...  (one sub-directory per calendar day)
```

- **Provider**: Danish Maritime Authority (Søfartsstyrelsen)
- **Format**: CSV, one file per day, naming `aisdk-YYYY-MM-DD.csv`
- **Coverage**: All AIS signals received by Danish coastal stations
  (primary region: North Sea / Baltic / Kattegat / Skagerrak)
- **Timestamp format**: `DD/MM/YYYY HH:MM:SS` (UTC)

### Glob pattern for multi-day loading
```python
from pathlib import Path
import pandas as pd

DATA_ROOT = Path("./data")
files = sorted(DATA_ROOT.glob("*/aisdk-*.csv"))  # all days
# or filtered:
files = sorted(DATA_ROOT.glob("2024-03-*/aisdk-*.csv"))
```

---

## DATA SCHEMA — FULL COLUMN SPECIFICATION

| # | Column | Raw Type | Description | Valid Values / Notes |
|---|--------|----------|-------------|----------------------|
| 1 | `# Timestamp` | str | UTC signal reception time | `DD/MM/YYYY HH:MM:SS` |
| 2 | `Type of mobile` | str | AIS transponder class | `Class A`, `Class B`, `Base Station`, `AtoN` |
| 3 | `MMSI` | int64 | Maritime Mobile Service Identity | 9-digit; see MMSI type table |
| 4 | `Latitude` | float64 | WGS-84 latitude (°N) | −90 to 90; 91.0 = not available |
| 5 | `Longitude` | float64 | WGS-84 longitude (°E) | −180 to 180; 181.0 = not available |
| 6 | `Navigational status` | str | Navigation state (text) | See nav-status table below |
| 7 | `ROT` | float64 | Rate of Turn (°/min) | −127 to 127; NaN = not available |
| 8 | `SOG` | float64 | Speed Over Ground (knots) | 0–102.2; 102.3 = not available |
| 9 | `COG` | float64 | Course Over Ground (°) | 0–359.9; 360.0 = not available |
| 10 | `Heading` | float64 | True Heading (°) | 0–359; 511 = not available |
| 11 | `IMO` | str | IMO number or `"Unknown"` | `"Unknown"` when not transmitted |
| 12 | `Callsign` | str | Radio call sign or `"Unknown"` | — |
| 13 | `Name` | str/NaN | Vessel name | Often NaN for Class B / Base Station |
| 14 | `Ship type` | str | Vessel type (text category) | See ship-type table below |
| 15 | `Cargo type` | float64 | Cargo/hazard sub-type code | Mostly NaN; numeric ITU sub-code |
| 16 | `Width` | float64 | Vessel beam (m) | NaN when not transmitted |
| 17 | `Length` | float64 | Vessel length overall (m) | NaN when not transmitted |
| 18 | `Type of position fixing device` | str | GNSS fix method | `Undefined`, `GPS`, `Surveyed`, `Internal`, `GLONASS`, `Galileo` |
| 19 | `Draught` | float64 | Keel depth (m) | 0–28; NaN when not transmitted |
| 20 | `Destination` | str | Reported destination port | `"Unknown"` when not transmitted |
| 21 | `ETA` | str/NaN | Estimated time of arrival | `DD/MM/YYYY HH:MM:SS`; NaN when not transmitted |
| 22 | `Data source type` | str | Signal medium | Always `"AIS"` in this dataset |
| 23 | `A` | float64 | Antenna offset from bow (m) | NaN when not transmitted |
| 24 | `B` | float64 | Antenna offset from stern (m) | NaN when not transmitted |
| 25 | `C` | float64 | Antenna offset from port (m) | NaN when not transmitted |
| 26 | `D` | float64 | Antenna offset from starboard (m) | NaN when not transmitted |

### Notes on `A`, `B`, `C`, `D`
These are GNSS antenna position offsets used to derive the vessel's reference point.
Ship total length ≈ A + B, total beam ≈ C + D. These enable dimension imputation
when `Length`/`Width` are NaN.

---

### Reference Tables

```python
# ── Navigational Status (text → integer code) ──────────────────────────────
NAV_STATUS_MAP = {
    "Under way using engine":        0,
    "At anchor":                     1,
    "Not under command":             2,   # distress / loss of control
    "Restricted maneuverability":    3,
    "Constrained by her draught":    4,
    "Moored":                        5,
    "Aground":                       6,
    "Engaged in fishing":            7,
    "Under way sailing":             8,
    "Reserved for HSC":              9,
    "Reserved for WIG":             10,
    "Power-driven vessel towing astern": 11,
    "Power-driven vessel pushing ahead": 12,
    "Reserved":                     13,
    "AIS-SART active":              14,
    "Unknown value":                15,  # treat as missing
}

# ── Ship Type (text → ITU integer code) ────────────────────────────────────
SHIP_TYPE_MAP = {
    "Not available":   0,
    "WIG":            20,
    "Fishing":        30,
    "Towing":         31,
    "Towing large":   32,
    "Dredging":       33,
    "Diving ops":     34,
    "Military":       35,   # ← KEY conflict indicator
    "Sailing":        36,
    "Pleasure":       37,
    "HSC":            40,
    "Pilot":          50,
    "SAR":            51,   # ← surge indicator during conflict
    "Tug":            52,
    "Port tender":    53,
    "Anti-pollution": 54,
    "Law enforce":    55,   # ← naval patrol indicator
    "Medical":        58,
    "Passenger":      60,
    "Cargo":          70,   # ← primary commercial target
    "Tanker":         80,   # ← energy security indicator
    "Other":          90,
    "Undefined":       0,
    "Reserved":        0,
}

# ── Type of Mobile → AIS Class ─────────────────────────────────────────────
MOBILE_CLASS_MAP = {
    "Class A":      "A",   # SOLAS mandatory (commercial vessels)
    "Class B":      "B",   # voluntary (small craft, leisure)
    "Base Station": "BS",  # coastal infrastructure (exclude from vessel analysis)
    "AtoN":         "AT",  # Aid to Navigation (buoys, lighthouses — exclude)
}

# ── MMSI Type Ranges ────────────────────────────────────────────────────────
MMSI_TYPE_MAP = {
    "vessel":        (200_000_000, 799_999_999),
    "coastal_bs":    (  0,          99_999_999),
    "group":         (970_000_000, 979_999_999),
    "sar_aircraft":  (111_000_000, 111_999_999),
    "aton":          (990_000_000, 999_999_999),
    "mob_device":    (972_000_000, 972_999_999),
    "search_rescue": (974_000_000, 974_999_999),
}
```

---

## REPOSITORY STRUCTURE

```
ais-conflict-intelligence/
├── CLAUDE.md                          # This file
├── README.md
├── requirements.txt
├── config/
│   └── settings.yaml                  # Paths, parameters, zone definitions
├── data/
│   ├── 2024-03-01/
│   │   └── aisdk-2024-03-01.csv       # Raw daily files (read-only)
│   ├── 2024-03-02/
│   │   └── aisdk-2024-03-02.csv
│   └── ...
├── data_external/
│   ├── acled_events.csv               # ACLED conflict event database
│   ├── gdelt_events.csv               # GDELT news events
│   ├── world_ports.csv                # World port coordinates & LOCODE
│   ├── un_locode.csv                  # UN/LOCODE port name → coordinate
│   └── geojson/
│       ├── conflict_zones.geojson     # Conflict zone polygons
│       ├── chokepoints.geojson        # Strategic strait polygons
│       └── eez_boundaries.geojson     # EEZ boundaries
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── loader.py                  # Multi-day file loader & merger
│   │   ├── cleaner.py                 # Data cleaning & validation
│   │   ├── feature_engineer.py        # Feature generation (A–F categories)
│   │   └── aggregator.py              # Grid × time-bucket aggregation
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── spatial_viz.py             # Geospatial: heatmaps, trajectories
│   │   ├── temporal_viz.py            # Time-series: traffic, composition
│   │   ├── statistical_viz.py         # Distributions, correlation, SHAP
│   │   └── conflict_overlay.py        # Conflict event overlay on maps
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── traffic_analyzer.py        # Volume & density analysis
│   │   ├── behavioral_analyzer.py     # Vessel behavior pattern analysis
│   │   ├── destination_analyzer.py    # Port call & destination analysis (NEW)
│   │   ├── rot_analyzer.py            # Rate-of-Turn anomaly analysis (NEW)
│   │   └── correlation_analyzer.py    # Granger, DiD, ITS, Event Study
│   └── models/
│       ├── __init__.py
│       ├── baseline.py                # ARIMA / Prophet baselines
│       ├── anomaly_model.py           # Isolation Forest, VAE
│       ├── conflict_predictor.py      # LSTM-Attention, TFT, XGBoost
│       └── evaluator.py              # Evaluation & reporting
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_preprocessing_validation.ipynb
│   ├── 03_visualization.ipynb
│   ├── 04_conflict_correlation.ipynb
│   └── 05_model_development.ipynb
├── outputs/
│   ├── processed/
│   │   ├── ais_clean.parquet
│   │   └── ais_features.parquet
│   ├── figures/
│   ├── tables/
│   ├── models/
│   └── reports/
└── scripts/
    ├── run_pipeline.sh
    └── generate_report.py
```

---

## PHASE 1 — PREPROCESSING (`src/preprocessing/`)

### Pipeline Diagram
```
./data/YYYY-MM-DD/aisdk-*.csv   (multi-day raw files)
         │
         ▼
  [loader.py]          ── lazy glob + chunked concat → raw DataFrame
         │
         ▼
  [cleaner.py]         ── MMSI filter, coordinate filter, sentinel → NaN,
                          timestamp parse, mobile type filter, dedup, impute
         │
         ▼
  [feature_engineer.py]── A. Kinematic  B. Geospatial  C. Behavioral
                          D. Destination/ETA  E. ROT  F. Conflict Labels
         │
         ▼
  [aggregator.py]      ── grid_cell × 6h bucket statistics
         │
         ▼
  outputs/processed/ais_clean.parquet
  outputs/processed/ais_features.parquet
```

---

### 1-0. `loader.py` — Multi-Day File Loader

```python
"""
AIS Multi-Day Loader
====================
Lazily loads and concatenates daily aisdk CSV files.
Supports date range filtering, chunked reading for large datasets,
and early column normalization.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Iterator
import logging

logger = logging.getLogger(__name__)

# ── Column rename map: raw aisdk → internal standard names ─────────────────
COLUMN_RENAME = {
    "# Timestamp":                    "timestamp",
    "Type of mobile":                 "mobile_type",
    "MMSI":                           "mmsi",
    "Latitude":                       "lat",
    "Longitude":                      "lon",
    "Navigational status":            "nav_status",
    "ROT":                            "rot",
    "SOG":                            "sog",
    "COG":                            "cog",
    "Heading":                        "heading",
    "IMO":                            "imo",
    "Callsign":                       "callsign",
    "Name":                           "vessel_name",
    "Ship type":                      "ship_type",
    "Cargo type":                     "cargo_type",
    "Width":                          "width",
    "Length":                         "length",
    "Type of position fixing device": "pos_fix_type",
    "Draught":                        "draught",
    "Destination":                    "destination",
    "ETA":                            "eta",
    "Data source type":               "data_source",
    "A":                              "ant_bow",
    "B":                              "ant_stern",
    "C":                              "ant_port",
    "D":                              "ant_starboard",
}

DTYPE_MAP = {
    "MMSI":        "int64",
    "Latitude":    "float32",
    "Longitude":   "float32",
    "ROT":         "float32",
    "SOG":         "float32",
    "COG":         "float32",
    "Heading":     "float32",
    "Cargo type":  "float32",
    "Width":       "float32",
    "Length":      "float32",
    "Draught":     "float32",
    "A":           "float32",
    "B":           "float32",
    "C":           "float32",
    "D":           "float32",
}


class AISDKLoader:
    """
    Loads aisdk daily CSV files from ./data/YYYY-MM-DD/aisdk-YYYY-MM-DD.csv.

    Parameters
    ----------
    data_root   : path to ./data directory
    date_start  : inclusive start date ("YYYY-MM-DD")
    date_end    : inclusive end date   ("YYYY-MM-DD")
    chunksize   : rows per chunk for large-file streaming (None = load all)
    """

    def __init__(
        self,
        data_root: str = "./data",
        date_start: str | None = None,
        date_end:   str | None = None,
        chunksize:  int | None = None,
    ):
        self.root       = Path(data_root)
        self.date_start = pd.Timestamp(date_start) if date_start else None
        self.date_end   = pd.Timestamp(date_end)   if date_end   else None
        self.chunksize  = chunksize

    # ------------------------------------------------------------------ utils
    def _get_files(self) -> list[Path]:
        """Glob all aisdk files, optionally filtered by date range."""
        files = sorted(self.root.glob("*/aisdk-*.csv"))
        if self.date_start or self.date_end:
            filtered = []
            for f in files:
                # Extract date from directory name: data/YYYY-MM-DD/aisdk-*.csv
                try:
                    d = pd.Timestamp(f.parent.name)
                except Exception:
                    continue
                if self.date_start and d < self.date_start:
                    continue
                if self.date_end and d > self.date_end:
                    continue
                filtered.append(f)
            files = filtered
        logger.info(f"Files matched: {len(files)}")
        return files

    def _read_file(self, path: Path) -> pd.DataFrame:
        """Read a single daily CSV with correct dtypes."""
        df = pd.read_csv(
            path,
            dtype=DTYPE_MAP,
            low_memory=False,
            na_values=["Unknown", "Undefined", ""],
        )
        df = df.rename(columns=COLUMN_RENAME)
        # Parse timestamps: format DD/MM/YYYY HH:MM:SS
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], format="%d/%m/%Y %H:%M:%S", utc=True, errors="coerce"
        )
        # Inject source date from directory name for traceability
        df["source_date"] = pd.Timestamp(path.parent.name).date()
        return df

    # ------------------------------------------------------------------ public
    def iter_chunks(self) -> Iterator[pd.DataFrame]:
        """Yield DataFrames file-by-file (memory-safe for large date ranges)."""
        for f in self._get_files():
            yield self._read_file(f)

    def load(self) -> pd.DataFrame:
        """Load and concatenate all matched files into a single DataFrame."""
        frames = [self._read_file(f) for f in self._get_files()]
        if not frames:
            raise FileNotFoundError(f"No aisdk files found in {self.root}")
        df = pd.concat(frames, ignore_index=True)
        logger.info(f"Total records loaded: {len(df):,}")
        return df
```

---

### 1-1. `cleaner.py` — Data Cleaning

```python
"""
AIS Data Cleaner (aisdk-specific)
===================================
Cleaning steps ordered by computational cost (cheapest first):

  1. Mobile type filter     — keep only Class A and Class B vessels
  2. MMSI validity          — 9-digit civilian range; flag specials
  3. Coordinate validity    — range + sentinel (91/181) removal
  4. Kinematic sentinels    — SOG=102.3, COG=360, Heading=511, ROT=NaN
  5. Timestamp validity     — future records, pre-2010 records
  6. Nav status encoding    — text → integer code
  7. Ship type encoding     — text → integer code
  8. Dimension imputation   — A+B → Length, C+D → Width when NaN
  9. Missing value strategy — per-column imputation
  10. Deduplication         — MMSI + timestamp
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

from src.preprocessing.loader import COLUMN_RENAME  # for reference

NAV_STATUS_MAP = {
    "Under way using engine":          0,
    "At anchor":                       1,
    "Not under command":               2,
    "Restricted maneuverability":      3,
    "Constrained by her draught":      4,
    "Moored":                          5,
    "Aground":                         6,
    "Engaged in fishing":              7,
    "Under way sailing":               8,
    "Reserved for HSC":                9,
    "Reserved for WIG":               10,
    "Power-driven vessel towing":     11,
    "Power-driven vessel pushing":    12,
    "Reserved":                       13,
    "AIS-SART active":                14,
    "Unknown value":                  15,
}

SHIP_TYPE_MAP = {
    "Fishing":    30, "Towing":      31, "Dredging":  33,
    "Diving ops": 34, "Military":    35, "Sailing":   36,
    "Pleasure":   37, "HSC":         40, "Pilot":     50,
    "SAR":        51, "Tug":         52, "Port tender":53,
    "Law enforce":55, "Medical":     58, "Passenger":  60,
    "Cargo":      70, "Tanker":      80, "Other":      90,
    "WIG":        20, "Undefined":    0, "Reserved":    0,
    "Not available": 0,
}

KEEP_MOBILE_TYPES = {"Class A", "Class B"}

SPECIAL_MMSI = {
    "coastal_bs":   (0,           99_999_999),
    "group":        (970_000_000, 979_999_999),
    "sar_aircraft": (111_000_000, 111_999_999),
    "aton":         (990_000_000, 999_999_999),
    "mob_device":   (972_000_000, 972_999_999),
}


class AISCleaner:

    def __init__(self, output_path: str = "outputs/processed/ais_clean.parquet"):
        self.output_path = Path(output_path)
        self.report: dict = {}

    # ── 1. Mobile type filter ───────────────────────────────────────────────
    def filter_mobile_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep Class A and Class B only (remove Base Station, AtoN)."""
        n = len(df)
        df = df[df["mobile_type"].isin(KEEP_MOBILE_TYPES)].copy()
        self.report["base_station_aton_removed"] = n - len(df)
        return df

    # ── 2. MMSI validity ────────────────────────────────────────────────────
    def filter_mmsi(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        for name, (lo, hi) in SPECIAL_MMSI.items():
            df.loc[df["mmsi"].between(lo, hi), "mmsi_special_type"] = name
        df = df[df["mmsi"].between(200_000_000, 799_999_999)].copy()
        self.report["mmsi_removed"] = n - len(df)
        return df

    # ── 3. Coordinate validity ──────────────────────────────────────────────
    def filter_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        df = df[
            df["lat"].between(-90.0, 90.0) &
            df["lon"].between(-180.0, 180.0) &
            (df["lat"] != 91.0) & (df["lon"] != 181.0) &
            df["lat"].notna() & df["lon"].notna()
        ].copy()
        self.report["coord_removed"] = n - len(df)
        return df

    # ── 4. Kinematic sentinels → NaN ────────────────────────────────────────
    def clean_kinematics(self, df: pd.DataFrame) -> pd.DataFrame:
        df["sog"]     = df["sog"].where(df["sog"]     < 102.3, np.nan)
        df["cog"]     = df["cog"].where(df["cog"]     < 360.0, np.nan)
        df["heading"] = df["heading"].where(df["heading"] < 511,  np.nan)
        # ROT: valid range −127 to +127 deg/min; flag implausible
        df["rot"]     = df["rot"].where(df["rot"].between(-127.0, 127.0), np.nan)
        df["sog_implausible"] = (df["sog"] > 50.0).astype("int8")
        return df

    # ── 5. Timestamp validity ───────────────────────────────────────────────
    def filter_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        now    = pd.Timestamp.now(tz="UTC")
        cutoff = pd.Timestamp("2010-01-01", tz="UTC")
        n = len(df)
        df = df[df["timestamp"].between(cutoff, now) & df["timestamp"].notna()].copy()
        self.report["timestamp_removed"] = n - len(df)
        return df.sort_values(["mmsi", "timestamp"]).reset_index(drop=True)

    # ── 6. Encode navigational status ───────────────────────────────────────
    def encode_nav_status(self, df: pd.DataFrame) -> pd.DataFrame:
        df["nav_status_code"] = (
            df["nav_status"].map(NAV_STATUS_MAP).fillna(15).astype("int8")
        )
        return df

    # ── 7. Encode ship type ─────────────────────────────────────────────────
    def encode_ship_type(self, df: pd.DataFrame) -> pd.DataFrame:
        df["ship_type_code"] = (
            df["ship_type"].map(SHIP_TYPE_MAP).fillna(0).astype("int16")
        )
        return df

    # ── 8. Dimension imputation from antenna offsets ─────────────────────────
    def impute_dimensions_from_antenna(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        AIS Message Type 5 encodes antenna reference position offsets:
          A = bow to antenna,  B = stern to antenna  → Length = A + B
          C = port to antenna, D = starboard to antenna → Width = C + D
        Use these to impute missing Length/Width.
        """
        computed_length = (df["ant_bow"].fillna(0) + df["ant_stern"].fillna(0))
        computed_width  = (df["ant_port"].fillna(0) + df["ant_starboard"].fillna(0))

        df["length"] = df["length"].fillna(
            computed_length.where(computed_length > 2.0)
        )
        df["width"] = df["width"].fillna(
            computed_width.where(computed_width > 1.0)
        )
        return df

    # ── 9. Missing value imputation ─────────────────────────────────────────
    def impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Per-column strategy:
          vessel_name / imo / callsign : fillna("UNKNOWN") already from loader
          ship_type_code               : per-MMSI mode → 0 (Unknown)
          length / width / draught     : per-MMSI median → per-ship_type median
          destination / eta            : retain NaN (used in route analysis)
          nav_status_code              : 15 (Unknown) as default
        """
        for col in ["vessel_name", "imo", "callsign", "destination"]:
            df[col] = df[col].fillna("UNKNOWN")

        # Ship type: per-MMSI mode
        type_mode = df.groupby("mmsi")["ship_type_code"].transform(
            lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else 0)
        )
        df["ship_type_code"] = df["ship_type_code"].fillna(type_mode).fillna(0)

        # Dimensions: per-MMSI median → per ship_type median
        for col in ["length", "width", "draught"]:
            df[col] = df.groupby("mmsi")[col].transform(lambda x: x.fillna(x.median()))
            df[col] = df[col].fillna(
                df.groupby("ship_type_code")[col].transform("median")
            )

        # Parse ETA as datetime (reuse timestamp format)
        df["eta_dt"] = pd.to_datetime(
            df["eta"], format="%d/%m/%Y %H:%M:%S", utc=True, errors="coerce"
        )
        return df

    # ── 10. Deduplication ───────────────────────────────────────────────────
    def deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        df = df.drop_duplicates(subset=["mmsi", "timestamp"], keep="first")
        self.report["duplicates_removed"] = n - len(df)
        return df

    # ── Pipeline runner ─────────────────────────────────────────────────────
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Input: {len(df):,} records")
        df = self.filter_mobile_type(df)
        df = self.filter_mmsi(df)
        df = self.filter_coordinates(df)
        df = self.clean_kinematics(df)
        df = self.filter_timestamps(df)
        df = self.encode_nav_status(df)
        df = self.encode_ship_type(df)
        df = self.impute_dimensions_from_antenna(df)
        df = self.impute_missing(df)
        df = self.deduplicate(df)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.output_path, index=False, compression="snappy")
        logger.info(f"Clean records: {len(df):,} → {self.output_path}")
        logger.info(f"Report: {self.report}")
        return df
```

---

### 1-2. `feature_engineer.py` — Feature Generation

```python
"""
AIS Feature Engineer (aisdk-specific)
======================================
Six feature categories:

  A. Kinematic Features       — speed, course, heading, maneuver dynamics
  B. Rate-of-Turn Features    — ROT-based maneuver classification (NEW)
  C. Geospatial Features      — conflict zone membership, chokepoint distance
  D. Behavioral Features      — rolling window irregularity metrics
  E. Destination/ETA Features — port call analysis & ETA deviation (NEW)
  F. Temporal Aggregation     — grid × 6h bucket statistics
  G. Conflict Labels          — binary label + regression target
"""
import pandas as pd
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class AISFeatureEngineer:

    CONFLICT_ZONES = {
        "black_sea":       {"bbox": [27.0, 40.5, 41.0, 46.8], "conflict": "ukraine_war"},
        "azov_sea":        {"bbox": [33.5, 45.0, 39.5, 47.5], "conflict": "ukraine_war"},
        "kerch_strait":    {"bbox": [36.4, 45.1, 36.8, 45.5], "conflict": "ukraine_war"},
        "red_sea":         {"bbox": [32.0, 12.0, 43.5, 30.0], "conflict": "houthi_crisis"},
        "bab_el_mandeb":   {"bbox": [43.0, 11.5, 45.0, 12.5], "conflict": "houthi_crisis"},
        "taiwan_strait":   {"bbox": [119.0, 22.0, 122.0, 26.0], "conflict": "taiwan_tension"},
        "south_china_sea": {"bbox": [109.0,  3.0, 121.0, 22.0], "conflict": "scs_dispute"},
        "strait_hormuz":   {"bbox": [56.0,  25.5,  59.5, 27.0], "conflict": "iran_tension"},
        "north_sea":       {"bbox": [−5.0,  51.0,  10.0, 58.0], "conflict": "none"},  # control zone
        "baltic_sea":      {"bbox": [10.0,  54.0,  30.0, 65.0], "conflict": "none"},  # control zone
    }

    CHOKEPOINTS = {
        "hormuz":     (56.50, 26.50),
        "malacca":    (103.80, 1.20),
        "bab_mandeb": (43.40, 12.50),
        "suez":       (32.50, 30.70),
        "panama":     (-79.90, 9.00),
        "gibraltar":  (-5.40, 36.00),
        "dover":      (1.30, 51.00),
        "danish_straits": (12.60, 55.70),  # relevant to aisdk coverage
        "skagerrak":  (9.50, 57.80),
    }

    # ── A. Kinematic Features ───────────────────────────────────────────────
    def add_kinematic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        speed_category    : anchored / drifting / slow / cruising / fast
        delta_sog         : |SOG_t − SOG_{t−1}| per MMSI
        delta_cog         : COG change, 360° wrap-corrected
        time_diff_sec     : seconds since previous fix
        turning_rate_cog  : delta_cog / time_diff_sec  (deg/s from COG)
        is_dark_ship      : AIS gap > 6 hours (flag: possible deliberate silence)
        sog_z_score       : per-MMSI standardized speed (baseline deviation)
        moored_flag       : nav_status_code == 5
        fishing_flag      : nav_status_code == 7
        """
        df = df.sort_values(["mmsi", "timestamp"]).copy()

        df["speed_category"] = pd.cut(
            df["sog"],
            bins=[-0.1, 0.5, 3.0, 8.0, 15.0, 102.2],
            labels=["anchored", "drifting", "slow", "cruising", "fast"],
        )

        grp = df.groupby("mmsi", sort=False)
        df["delta_sog"]     = grp["sog"].diff().abs()
        raw_dcog            = grp["cog"].diff().abs()
        df["delta_cog"]     = raw_dcog.apply(
            lambda x: min(x, 360.0 - x) if pd.notna(x) else np.nan
        )
        df["time_diff_sec"] = grp["timestamp"].diff().dt.total_seconds()
        df["turning_rate_cog"] = (
            df["delta_cog"] / df["time_diff_sec"].replace(0, np.nan)
        )
        df["is_dark_ship"]  = (df["time_diff_sec"] > 21_600).astype("int8")
        df["sog_z_score"]   = grp["sog"].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-6)
        )
        df["moored_flag"]   = (df["nav_status_code"] == 5).astype("int8")
        df["fishing_flag"]  = (df["nav_status_code"] == 7).astype("int8")
        return df

    # ── B. Rate-of-Turn Features (NEW — aisdk has ROT column) ───────────────
    def add_rot_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ROT (deg/min) is available in aisdk — critical for maneuver classification.

        rot_category     : "hard_port" | "port" | "steady" | "starboard" | "hard_starboard"
        rot_abs          : |ROT|
        rot_spike        : ROT changes >20 deg/min between consecutive fixes
        evasive_maneuver : high ROT (>10 deg/min) + high SOG (>5 kn) in conflict zone
        rot_vs_cog_consistency : |rot| > 5 but delta_cog < 1° → sensor anomaly flag
        """
        df["rot_abs"] = df["rot"].abs()

        df["rot_category"] = pd.cut(
            df["rot"],
            bins=[-128, -20, -5, 5, 20, 128],
            labels=["hard_port", "port", "steady", "starboard", "hard_starboard"],
        )

        grp = df.groupby("mmsi", sort=False)
        df["rot_spike"] = (
            grp["rot"].diff().abs() > 20.0
        ).astype("int8")

        df["evasive_maneuver"] = (
            (df["rot_abs"] > 10.0) &
            (df["sog"] > 5.0) &
            df["in_conflict_zone"]   # requires geospatial features run first
        ).astype("int8")

        # Sensor consistency: large ROT but course barely changed
        df["rot_cog_inconsistent"] = (
            (df["rot_abs"] > 5.0) & (df["delta_cog"] < 1.0)
        ).astype("int8")

        return df

    # ── C. Geospatial Features ───────────────────────────────────────────────
    def add_geospatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        grid_cell         : 0.5° × 0.5° cell ID
        in_conflict_zone  : bool — inside any conflict zone
        conflict_zone_name: zone label or "none"
        is_control_zone   : bool — inside a designated control zone
        dist_{cp}_km      : Haversine distance to each strategic chokepoint
        """
        df["grid_lat"]  = (df["lat"] // 0.5) * 0.5
        df["grid_lon"]  = (df["lon"] // 0.5) * 0.5
        df["grid_cell"] = df["grid_lat"].astype(str) + "_" + df["grid_lon"].astype(str)

        df["in_conflict_zone"]   = False
        df["conflict_zone_name"] = "none"
        df["is_control_zone"]    = False
        for zone, info in self.CONFLICT_ZONES.items():
            b = info["bbox"]
            mask = df["lon"].between(b[0], b[2]) & df["lat"].between(b[1], b[3])
            if info["conflict"] == "none":
                df.loc[mask, "is_control_zone"] = True
            else:
                df.loc[mask, "in_conflict_zone"]   = True
                df.loc[mask, "conflict_zone_name"] = zone

        for name, (cp_lon, cp_lat) in self.CHOKEPOINTS.items():
            df[f"dist_{name}_km"] = self._haversine(df["lat"], df["lon"], cp_lat, cp_lon)

        return df

    # ── D. Behavioral Features ────────────────────────────────────────────────
    def add_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        rolling_sog_mean_12h : 12-hour rolling mean SOG per MMSI
        rolling_sog_std_12h  : 12-hour rolling std SOG per MMSI
        rolling_rot_abs_12h  : 12-hour rolling mean |ROT| per MMSI
        route_entropy        : Shannon entropy of daily COG distribution (36 bins)
        loitering_flag       : slow + frequent turns inside conflict zone
        zig_zag_index        : COG sign reversal count (10-point window)
        not_under_command    : nav_status_code == 2 flag
        """
        df = df.sort_values(["mmsi", "timestamp"]).set_index("timestamp")

        grp = df.groupby("mmsi")
        for col, label in [("sog", "sog"), ("rot_abs", "rot_abs")]:
            roll = grp[col].rolling("12H", min_periods=3)
            df[f"rolling_{label}_mean_12h"] = roll.mean().reset_index(level=0, drop=True)
            df[f"rolling_{label}_std_12h"]  = roll.std().reset_index(level=0, drop=True)

        df = df.reset_index()

        def _entropy(s: pd.Series) -> float:
            bins = pd.cut(s.dropna(), bins=36, labels=False)
            p = bins.value_counts(normalize=True) + 1e-10
            return float(stats.entropy(p))

        df["_date"] = df["timestamp"].dt.date
        ent = (
            df.groupby(["mmsi", "_date"])["cog"]
            .apply(_entropy)
            .reset_index()
            .rename(columns={"cog": "route_entropy"})
        )
        df = df.merge(ent, on=["mmsi", "_date"], how="left").drop(columns="_date")

        df["loitering_flag"] = (
            (df["sog"] < 3.0) &
            (df["delta_cog"] > 45.0) &
            df["in_conflict_zone"]
        ).astype("int8")

        df["_cog_sign"]     = np.sign(df["delta_cog"].fillna(0))
        df["zig_zag_index"] = (
            df.groupby("mmsi")["_cog_sign"]
            .transform(lambda x: (x != x.shift()).rolling(10, min_periods=3).sum())
        )
        df.drop(columns="_cog_sign", inplace=True)

        df["not_under_command"] = (df["nav_status_code"] == 2).astype("int8")
        return df

    # ── E. Destination / ETA Features (NEW — aisdk has Destination & ETA) ───
    def add_destination_features(
        self, df: pd.DataFrame, ports_path: str
    ) -> pd.DataFrame:
        """
        Exploit the `destination` and `eta_dt` columns unique to aisdk.

        dest_is_conflict_port     : declared destination is a port in a conflict zone
        dest_changed              : vessel changed destination between consecutive fixes
        eta_hours_remaining       : hours until ETA from current fix
        eta_implausible           : ETA is in the past (spoofing / stale data)
        destination_count_24h     : unique destinations declared in 24h (high = suspicious)
        """
        ports = pd.read_csv(ports_path)  # world_ports.csv: port_name, lat, lon, zone
        conflict_ports = set(
            ports.loc[ports["conflict_zone"] != "none", "locode"].str.upper()
        )

        df["dest_upper"] = df["destination"].str.upper().str.strip()
        df["dest_is_conflict_port"] = (
            df["dest_upper"].isin(conflict_ports)
        ).astype("int8")

        df["dest_changed"] = (
            df.groupby("mmsi")["destination"]
            .transform(lambda x: (x != x.shift()).astype("int8"))
        )

        df["eta_hours_remaining"] = (
            (df["eta_dt"] - df["timestamp"]).dt.total_seconds() / 3600
        )
        df["eta_implausible"] = (df["eta_hours_remaining"] < -1.0).astype("int8")

        df["_date"] = df["timestamp"].dt.date
        dest_count = (
            df.groupby(["mmsi", "_date"])["destination"]
            .transform("nunique")
        )
        df["destination_count_24h"] = dest_count
        df.drop(columns=["_date", "dest_upper"], inplace=True)
        return df

    # ── F. Temporal Aggregation ───────────────────────────────────────────────
    def add_temporal_aggregation(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Unit: grid_cell × 6-hour time bucket

        traffic_count          : unique MMSIs
        class_a_ratio          : fraction Class A (SOLAS mandatory)
        dark_ship_ratio        : fraction with AIS gap > 6h
        military_ratio         : fraction ship_type_code == 35
        fishing_ratio          : fraction ship_type_code == 30
        tanker_ratio           : fraction ship_type_code in {80–89}
        sar_count              : SAR vessel count
        mean_sog / std_sog     : speed statistics
        mean_rot_abs           : mean |ROT| (maneuver intensity)
        loitering_density      : loitering flag sum
        not_under_command_count: distress / loss-of-control count
        evasive_count          : evasive maneuver flag sum
        dest_conflict_count    : vessels declaring conflict port as destination
        """
        df["time_bucket"] = df["timestamp"].dt.floor("6H")
        agg = (
            df.groupby(["grid_cell", "time_bucket"])
            .agg(
                traffic_count          =("mmsi",              "nunique"),
                class_a_count          =("mobile_type",       lambda x: (x == "Class A").sum()),
                dark_ship_count        =("is_dark_ship",       "sum"),
                military_count         =("ship_type_code",     lambda x: (x == 35).sum()),
                fishing_count          =("ship_type_code",     lambda x: (x == 30).sum()),
                tanker_count           =("ship_type_code",     lambda x: x.isin(range(80, 90)).sum()),
                cargo_count            =("ship_type_code",     lambda x: x.isin(range(70, 80)).sum()),
                sar_count              =("ship_type_code",     lambda x: (x == 51).sum()),
                mean_sog               =("sog",                "mean"),
                std_sog                =("sog",                "std"),
                mean_rot_abs           =("rot_abs",            "mean"),
                loitering_density      =("loitering_flag",     "sum"),
                not_under_command_count=("not_under_command",  "sum"),
                evasive_count          =("evasive_maneuver",   "sum"),
                dest_conflict_count    =("dest_is_conflict_port","sum"),
            )
            .reset_index()
        )
        denom = agg["traffic_count"].clip(lower=1)
        for col, cnt in [
            ("class_a_ratio",  "class_a_count"),
            ("dark_ship_ratio","dark_ship_count"),
            ("military_ratio", "military_count"),
            ("fishing_ratio",  "fishing_count"),
            ("tanker_ratio",   "tanker_count"),
        ]:
            agg[col] = agg[cnt] / denom

        self.agg_df = agg
        return df, agg

    # ── G. Conflict Labels ────────────────────────────────────────────────────
    def add_conflict_labels(
        self, df: pd.DataFrame, conflict_events_path: str
    ) -> pd.DataFrame:
        """
        conflict_label     : 1 if conflict event occurs within 30 days in same zone
        days_to_conflict   : signed days to nearest event (regression target)
        conflict_intensity : ACLED fatality count (continuous severity proxy)
        """
        events = pd.read_csv(conflict_events_path, parse_dates=["event_date"])
        df["conflict_label"]     = 0
        df["days_to_conflict"]   = np.nan
        df["conflict_intensity"] = 0.0

        for _, ev in events.iterrows():
            zone_mask = df["conflict_zone_name"] == ev.get("zone", "")
            day_diff  = (
                ev["event_date"] - df["timestamp"].dt.tz_localize(None)
            ).dt.days
            match = zone_mask & day_diff.between(-7, 30)
            df.loc[match, "conflict_label"]     = 1
            df.loc[match, "days_to_conflict"]   = day_diff[match]
            df.loc[match, "conflict_intensity"] = ev.get("fatalities", 0)
        return df

    # ── util ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _haversine(lat1, lon1, lat2: float, lon2: float) -> pd.Series:
        R  = 6_371.0
        p1 = np.radians(lat1); p2 = np.radians(lat2)
        dp = np.radians(lat2 - lat1); dl = np.radians(lon2 - lon1)
        a  = np.sin(dp / 2)**2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2)**2
        return R * 2 * np.arcsin(np.sqrt(a))

    def run(
        self, df: pd.DataFrame,
        conflict_events_path: str,
        ports_path: str,
    ) -> pd.DataFrame:
        logger.info("A. Kinematic features...")
        df = self.add_kinematic_features(df)
        logger.info("C. Geospatial features...")       # C before B (B needs in_conflict_zone)
        df = self.add_geospatial_features(df)
        logger.info("B. ROT features...")
        df = self.add_rot_features(df)
        logger.info("D. Behavioral features...")
        df = self.add_behavioral_features(df)
        logger.info("E. Destination/ETA features...")
        df = self.add_destination_features(df, ports_path)
        logger.info("F. Temporal aggregation...")
        df, _ = self.add_temporal_aggregation(df)
        logger.info("G. Conflict labels...")
        df = self.add_conflict_labels(df, conflict_events_path)
        return df
```

---

## PHASE 2 — VISUALIZATION (`src/visualization/`)

### `spatial_viz.py`
```python
"""
Outputs → outputs/figures/spatial/

Figures:
  1. Global density heatmap — Folium HeatMapWithTime, 6-hour steps
     Conflict zone polygons + chokepoint markers overlaid
  2. Per-MMSI trajectory — SOG color-mapped polyline
     ROT encoded as arrow direction overlaid (NEW)
  3. Dark ship cluster map — DBSCAN inside conflict zones
  4. Destination flow map — origin → destination Sankey/arc map (NEW)
     highlight vessels declaring conflict-zone ports
  5. AtoN / Base Station coverage map — aisdk receiver footprint
"""

### `temporal_viz.py`
"""
Outputs → outputs/figures/temporal/

Figures:
  1. Daily traffic volume per conflict zone — line chart + conflict onset axvlines
     95% CI shading pre/post
  2. Vessel type composition — stacked area chart (Class A ratio, Military, SAR)
  3. Speed distribution change — monthly violin plots per zone
  4. Dark ship ratio time series — 7-day rolling average
  5. Mean |ROT| time series — maneuver intensity over time (NEW)
  6. Destination diversity index — unique destinations per zone per day (NEW)
  7. CCF: AIS indicators vs. conflict intensity — ±30-day lead/lag
"""

### `statistical_viz.py`
"""
Outputs → outputs/figures/statistical/

Figures:
  1. Pre/post conflict distributions — KDE + boxplot for each feature
  2. Spearman correlation heatmap — AIS features × conflict intensity
  3. Feature importance bar chart — Random Forest / XGBoost
  4. ROC-AUC comparison — LSTM / TFT / XGBoost / Prophet baseline
  5. Precision–Recall curves — imbalanced label evaluation
  6. SHAP summary — top-20 features, bee-swarm plot
  7. Confusion matrices — T+7, T+14, T+30 horizons per zone
  8. ROT distribution pre/post — violin plots (NEW)
  9. Destination change rate — pre/post conflict comparison (NEW)
"""
```

---

## PHASE 3 — ANALYSIS (`src/analysis/`)

### `correlation_analyzer.py`
```python
"""
Statistical Conflict Correlation Framework
==========================================
Methods:
  1. Granger Causality Test   — AIS anomaly index → conflict intensity
  2. Cross-Correlation (CCF)  — optimal lead window (days)
  3. Difference-in-Differences — treatment (conflict) vs. control (N. Sea / Baltic)
  4. Event Study              — ±30-day ATV (Abnormal Traffic Volume)
  5. Interrupted Time Series  — OLS level + slope change at conflict onset

Key hypothesis tests per conflict zone:
  H1: Traffic volume declines significantly post-conflict onset
  H2: Dark ship ratio increases before conflict onset  (Granger lead)
  H3: Military/SAR vessel ratio increases pre-conflict
  H4: Mean |ROT| increases pre-conflict (heightened maneuver intensity)
  H5: Destination diversity drops (rerouting away from conflict zones)
"""

CONFLICT_ONSET = {
    "ukraine_war":      "2022-02-24",
    "houthi_crisis":    "2023-11-19",
    "pla_taiwan_drill": "2022-08-04",
    "kerch_bridge":     "2022-10-08",
}

CONTROL_ZONES = ["north_sea", "baltic_sea"]  # DiD comparators
```

---

## PHASE 4 — MODELS (`src/models/`)

### Feature Set Summary for Modeling
```python
# Primary feature vector (per grid_cell × 6h bucket)
FEATURES_AGGREGATE = [
    "traffic_count",
    "class_a_ratio",
    "dark_ship_ratio",
    "military_ratio",
    "fishing_ratio",
    "tanker_ratio",
    "sar_count",
    "mean_sog", "std_sog",
    "mean_rot_abs",          # NEW — from ROT column
    "loitering_density",
    "not_under_command_count",
    "evasive_count",
    "dest_conflict_count",   # NEW — from Destination column
    "destination_count_24h", # NEW — destination diversity
    "route_entropy",
    "zig_zag_index",
]

# Per-vessel sequence features for LSTM (30-day lookback window)
FEATURES_SEQUENCE = [
    "sog", "cog", "heading",
    "rot", "rot_abs",        # NEW
    "delta_sog", "delta_cog",
    "turning_rate_cog",
    "is_dark_ship",
    "sog_z_score",
    "rolling_sog_mean_12h", "rolling_sog_std_12h",
    "rolling_rot_abs_mean_12h",  # NEW
    "route_entropy",
    "zig_zag_index",
    "not_under_command",
    "evasive_maneuver",      # NEW
    "dest_is_conflict_port", # NEW
    "eta_hours_remaining",   # NEW
    "dist_hormuz_km", "dist_bab_mandeb_km", "dist_suez_km",
    "dist_danish_straits_km", "dist_skagerrak_km",
]
```

### `anomaly_model.py`
```python
"""
Unsupervised Maritime Anomaly Detection
=======================================
Models:
  1. Isolation Forest       — multivariate point anomaly score
  2. Variational Autoencoder — reconstruction error on normal pattern
  3. DBSCAN                  — spatial density cluster anomaly
  4. Local Outlier Factor    — neighborhood density deviation

Anomaly type taxonomy:
  "dark_ship"         — is_dark_ship == 1 (AIS gap > 6h)
  "evasive_maneuver"  — high ROT + high SOG in conflict zone (NEW)
  "loitering"         — slow + frequent turns in conflict zone
  "zig_zag"           — high zig_zag_index
  "density_surge"     — traffic_count spike vs. rolling baseline
  "destination_spoof" — eta_implausible or dest_changed frequently (NEW)
  "speed_spike"       — sog_z_score > 3σ
"""
```

### `conflict_predictor.py`
```python
"""
Conflict Prediction Model
=========================
Predicts P(conflict within T+N days) per grid_cell × time bucket.

Architectures:
  1. Bidirectional LSTM + Multi-Head Attention  (primary temporal)
  2. Temporal Fusion Transformer (TFT)          (static + dynamic inputs)
  3. XGBoost                                    (interpretable baseline)
  4. ARIMA / Prophet                            (univariate statistical baseline)

Training:
  Target   : conflict_label (binary); days_to_conflict (regression)
  Horizons : T+3, T+7, T+14, T+30 days
  Imbalance: SMOTE + focal loss (γ=2)
  Split    : temporal — train ≤ 2023-06, val 2023-07–09, test 2023-10+

Evaluation:
  AUROC, AUPRC, F2-Score, Mean Lead Time, False Alarm Rate
"""
import torch, torch.nn as nn


class ConflictLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.attention = nn.MultiheadAttention(
            hidden_dim * 2, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _  = self.lstm(x)
        attn, _ = self.attention(out, out, out)
        return self.head(self.norm(out + attn).mean(dim=1)).squeeze(-1)


class FocalBCELoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma; self.alpha = alpha

    def forward(self, pred, target):
        bce = nn.functional.binary_cross_entropy(pred, target, reduction="none")
        pt  = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()
```

---

## EXTERNAL DATA SOURCES

| Dataset | Source | URL | Purpose |
|---------|--------|-----|---------|
| ACLED conflict events | ACLED | acleddata.com | Conflict labels & intensity |
| GDELT news events | GDELT 2.0 | gdeltproject.org | Conflict intensity proxy |
| EEZ boundaries | MarineRegions | marineregions.org | Jurisdiction features |
| World port index | NGA (US) | msi.nga.mil | Destination port geolocation |
| UN/LOCODE | UNECE | unece.org/locode | Destination name → coordinates |
| Sea area polygons | OpenSeaMap | openseamap.org | Conflict zone definition |
| Sanctioned vessels | OFAC SDN | sanctionssearch.ofac.treas.gov | Sanctions evasion detection |
| Extended AIS history | MarineTraffic / AISHub | marinetraffic.com | Pre-2024 time series |

---

## EXECUTION COMMANDS

```bash
# Environment
conda create -n mcis python=3.11 -y && conda activate mcis
pip install -r requirements.txt

# Step 1 — Load & clean
python -m src.preprocessing.loader \
    --data-root ./data \
    --date-start 2024-03-01 \
    --date-end   2024-08-30 \
    --output     outputs/processed/ais_raw_march.parquet

python -m src.preprocessing.cleaner \
    --input  outputs/processed/ais_raw_march.parquet \
    --output outputs/processed/ais_clean.parquet

# Step 2 — Feature engineering
python -m src.preprocessing.feature_engineer \
    --input           outputs/processed/ais_clean.parquet \
    --conflict-events data_external/acled_events.csv \
    --ports           data_external/world_ports.csv \
    --output          outputs/processed/ais_features.parquet

# Step 3 — Visualization
python -m src.visualization.spatial_viz \
    --input outputs/processed/ais_features.parquet \
    --output-dir outputs/figures/

# Step 4 — Correlation analysis
python -m src.analysis.correlation_analyzer \
    --input  outputs/processed/ais_features.parquet \
    --output outputs/tables/

# Step 5–6 — Model training
python -m src.models.anomaly_model \
    --input outputs/processed/ais_features.parquet \
    --output outputs/models/anomaly/

python -m src.models.conflict_predictor \
    --input outputs/processed/ais_features.parquet \
    --mode train --output outputs/models/predictor/

# Full pipeline
bash scripts/run_pipeline.sh
```

---

## REQUIREMENTS

```
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.11.0
scikit-learn>=1.3.0
torch>=2.0.0
statsmodels>=0.14.0
geopandas>=0.13.0
shapely>=2.0.0
folium>=0.14.0
plotly>=5.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
pyarrow>=12.0.0
xgboost>=1.7.0
lightgbm>=4.0.0
shap>=0.42.0
imbalanced-learn>=0.11.0
prophet>=1.1.4
pytorch-forecasting>=1.0.0
openpyxl>=3.1.0
jupyter>=1.0.0
tqdm>=4.65.0
pyyaml>=6.0
```

---

## PAPER STRUCTURE

```
Title:
  "Maritime Traffic Anomaly Detection as a Precursor to Armed Conflict:
   Evidence from AIS Data in Global Hotspots (2022–2024)"

Abstract

1.  Introduction
    1.1  Motivation — AIS as a conflict early-warning signal
    1.2  Research questions and hypotheses (H1–H5)
    1.3  Dataset & contributions (aisdk + ACLED)

2.  Background and Related Work
    2.1  AIS system architecture and data characteristics
    2.2  AIS limitations: spoofing, dark shipping, Class B gaps
    2.3  Prior work: maritime anomaly detection
    2.4  Prior work: conflict early-warning systems

3.  Data and Methodology
    3.1  aisdk multi-day data pipeline
    3.2  Conflict event data (ACLED integration)
    3.3  Feature engineering — six categories including ROT & Destination
    3.4  Analytical framework (Granger, DiD, ITS, Event Study)

4.  Empirical Results
    4.1  Black Sea (Russia–Ukraine War) — traffic collapse, dark ship surge
    4.2  Red Sea (Houthi Crisis) — tanker rerouting, destination shift
    4.3  Taiwan Strait — PLA exercise maneuver intensity (ROT)
    4.4  Leading indicator analysis — Granger test, optimal lead-lag

5.  Predictive Modeling
    5.1  Model comparison — LSTM vs. TFT vs. XGBoost
    5.2  Zone-level performance (AUROC, F2, Lead Time)
    5.3  SHAP analysis — ROT & dark_ship_ratio as dominant predictors
    5.4  Horizon sensitivity — T+3 to T+30

6.  Discussion
    6.1  AIS data limitations and validity threats
    6.2  Operational implications for maritime conflict early warning

7.  Conclusion

References
Appendix A — Supplementary Figures
Appendix B — Full Statistical Tables
Appendix C — aisdk Schema Documentation
```

---

## AGENTIC EXECUTION PLAN (Claude Code)

```
[Task 1]  src/preprocessing/loader.py   + tests/test_loader.py
          Multi-day glob + rename + timestamp parse
          → outputs/processed/ais_raw_march.parquet

[Task 2]  src/preprocessing/cleaner.py  + tests/test_cleaner.py
          Mobile filter, MMSI, coord, kinematics, nav_status & ship_type encode,
          antenna-offset dimension imputation, dedup
          → outputs/processed/ais_clean.parquet

[Task 3]  src/preprocessing/feature_engineer.py  + tests/test_features.py
          A–G feature categories incl. ROT & Destination (aisdk-specific)
          → outputs/processed/ais_features.parquet

[Task 4]  notebooks/01_EDA.ipynb
          → outputs/figures/eda/

[Task 5]  src/visualization/  (spatial + temporal + statistical)
          → outputs/figures/  (300 dpi PNG)

[Task 6]  src/analysis/correlation_analyzer.py
          Granger (H1–H5) + DiD (N.Sea control) + ITS + Event Study
          → outputs/tables/

[Task 7]  src/models/anomaly_model.py
          IF + VAE; anomaly_type taxonomy
          → outputs/models/anomaly/

[Task 8]  src/models/conflict_predictor.py
          LSTM-Attention + XGBoost; T+7 / T+30 AUC
          → outputs/models/predictor/

[Task 9]  scripts/generate_report.py
          → outputs/reports/mcis_final_report.pdf
```

---

## CODING CONVENTIONS

```python
# 1. Type hints on all public functions
def process(df: pd.DataFrame, config: dict) -> pd.DataFrame: ...

# 2. Logging — no bare print() inside src/
import logging; logger = logging.getLogger(__name__)

# 3. No hardcoded values — config/settings.yaml for all paths & params

# 4. Raw data = READ-ONLY. Never write to ./data/

# 5. All intermediate data → Parquet (snappy compression)
df.to_parquet(path, index=False, compression="snappy")

# 6. Reproducibility
SEED = 42
import random, numpy as np, torch
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# 7. Publication figures
import matplotlib.pyplot as plt
plt.rcParams.update({
    "figure.dpi": 300, "figure.figsize": (10, 6),
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
})

# 8. Timestamp column name: "timestamp" (internal); parse format "%d/%m/%Y %H:%M:%S"
# 9. All stat outputs must include p-value, effect size, and confidence interval
# 10. Every module has a CLI entry point via argparse
# 11. tqdm progress bars for loops > 1000 iterations
```
