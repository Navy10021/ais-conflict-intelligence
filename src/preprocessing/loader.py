"""
AIS Multi-Day Loader
===================
Lazily loads and concatenates daily aisdk CSV files.
Supports date range filtering, chunked reading for large datasets,
and early column normalization.
"""
import pandas as pd
from pathlib import Path
from typing import Iterator, Optional
import logging
import argparse
import yaml

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
        date_start: Optional[str] = None,
        date_end:   Optional[str] = None,
        chunksize:  Optional[int] = None,
    ):
        self.root       = Path(data_root)
        self.date_start = pd.Timestamp(date_start) if date_start else None
        self.date_end   = pd.Timestamp(date_end)   if date_end   else None
        self.chunksize  = chunksize
        self._config_chunksize = self._load_chunksize_from_config()
        if self.chunksize is None:
            self.chunksize = self._config_chunksize

    def _load_chunksize_from_config(self) -> Optional[int]:
        config_path = Path("config/settings.yaml")
        if not config_path.exists():
            return None
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        value = config.get("preprocessing", {}).get("chunksize")
        return int(value) if value else None

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
        logger.info("Files matched: %d", len(files))
        return files

    def _read_file(self, path: Path) -> pd.DataFrame:
        """Read a single daily CSV with optional chunk streaming."""
        reader = pd.read_csv(
            path,
            low_memory=False,
            na_values=["Unknown", "Undefined", ""],
            chunksize=self.chunksize,
        )
        if self.chunksize:
            df = pd.concat(reader, ignore_index=True)
        else:
            df = reader
        
        # Apply dtype only to columns that exist and can be converted
        for col, dtype in DTYPE_MAP.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except (ValueError, TypeError):
                    logger.warning("Could not convert column %s to %s, keeping original", col, dtype)
        
        df = df.rename(columns=COLUMN_RENAME)

        # Fix string columns that might have numeric values
        for col in ["imo", "callsign", "vessel_name", "destination", "eta"]:
            if col in df.columns:
                df[col] = df[col].astype(str).replace("nan", "UNKNOWN")

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
        files = self._get_files()
        frames = [self._read_file(f) for f in files]
        if not frames:
            raise FileNotFoundError(f"No aisdk files found in {self.root}")
        df = pd.concat(frames, ignore_index=True)
        logger.info("Total records loaded: %d", len(df))
        return df


def main():
    parser = argparse.ArgumentParser(description="Load aisdk CSV files")
    parser.add_argument("--data-root", default="./data", help="Path to data directory")
    parser.add_argument("--date-start", help="Start date YYYY-MM-DD")
    parser.add_argument("--date-end", help="End date YYYY-MM-DD")
    parser.add_argument("--output", default="outputs/processed/ais_raw.parquet",
                        help="Output parquet file path")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    loader = AISDKLoader(
        data_root=args.data_root,
        date_start=args.date_start,
        date_end=args.date_end,
    )
    df = loader.load()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, compression="snappy")
    logger.info("Saved to %s", output_path)
    print(f"Loaded {len(df):,} records → {output_path}")


if __name__ == "__main__":
    main()
