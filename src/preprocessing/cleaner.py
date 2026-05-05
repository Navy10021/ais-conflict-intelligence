"""
AIS Data Cleaner (aisdk-specific)
===================================
Cleaning steps ordered by computational cost (cheapest first):

  1. Mobile type filter     ->keep only Class A and Class B vessels
  2. MMSI validity          ->9-digit civilian range; flag specials
  3. Coordinate validity    ->range + sentinel (91/181) removal
  4. Kinematic sentinels    ->SOG=102.3, COG=360, Heading=511, ROT=NaN
  5. Timestamp validity     ->future records, pre-2010 records
  6. Nav status encoding    ->text ->integer code
  7. Ship type encoding     ->text ->integer code
  8. Dimension imputation   ->A+B ->Length, C+D ->Width when NaN
  9. Missing value strategy ->per-column imputation
  10. Deduplication         ->MMSI + timestamp
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse

logger = logging.getLogger(__name__)

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
    "Power-driven vessel pushing":     12,
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

    # ?? 1. Mobile type filter ???????????????????????????????????????????????
    def filter_mobile_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep Class A and Class B only (remove Base Station, AtoN)."""
        n = len(df)
        df = df[df["mobile_type"].isin(KEEP_MOBILE_TYPES)].copy()
        self.report["base_station_aton_removed"] = n - len(df)
        logger.info("Mobile type filter: removed %d records", n - len(df))
        return df

    # ?? 2. MMSI validity ????????????????????????????????????????????????????
    def filter_mmsi(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        df = df[df["mmsi"].between(200_000_000, 799_999_999)].copy()
        self.report["mmsi_removed"] = n - len(df)
        logger.info("MMSI filter: removed %d records", n - len(df))
        return df

    # ?? 3. Coordinate validity ??????????????????????????????????????????????
    def filter_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        df = df[
            df["lat"].between(-90.0, 90.0) &
            df["lon"].between(-180.0, 180.0) &
            (df["lat"] != 91.0) & (df["lon"] != 181.0) &
            df["lat"].notna() & df["lon"].notna()
        ].copy()
        self.report["coord_removed"] = n - len(df)
        logger.info("Coordinate filter: removed %d records", n - len(df))
        return df

    # ?? 4. Kinematic sentinels ->NaN ????????????????????????????????????????
    def clean_kinematics(self, df: pd.DataFrame) -> pd.DataFrame:
        df["sog"]     = df["sog"].where(df["sog"]     < 102.3, np.nan)
        df["cog"]     = df["cog"].where(df["cog"]     < 360.0, np.nan)
        df["heading"] = df["heading"].where(df["heading"] < 511,  np.nan)
        # ROT: valid range ->27 to +127 deg/min; flag implausible
        df["rot"]     = df["rot"].where(df["rot"].between(-127.0, 127.0), np.nan)
        df["sog_implausible"] = (df["sog"] > 50.0).astype("int8")
        return df

    # ?? 5. Timestamp validity ???????????????????????????????????????????????
    def filter_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        # Handle both timezone-aware and naive timestamps
        if df["timestamp"].dt.tz is not None:
            now = pd.Timestamp.now(tz="UTC")
            cutoff = pd.Timestamp("2010-01-01", tz="UTC")
        else:
            now = pd.Timestamp.now()
            cutoff = pd.Timestamp("2010-01-01")
        n = len(df)
        df = df[df["timestamp"].between(cutoff, now) & df["timestamp"].notna()].copy()
        self.report["timestamp_removed"] = n - len(df)
        logger.info("Timestamp filter: removed %d records", n - len(df))
        return df.sort_values(["mmsi", "timestamp"]).reset_index(drop=True)

    # ?? 6. Encode navigational status ???????????????????????????????????????
    def encode_nav_status(self, df: pd.DataFrame) -> pd.DataFrame:
        df["nav_status_code"] = (
            df["nav_status"].map(NAV_STATUS_MAP).fillna(15).astype("int8")
        )
        return df

    # ?? 7. Encode ship type ?????????????????????????????????????????????????
    def encode_ship_type(self, df: pd.DataFrame) -> pd.DataFrame:
        df["ship_type_code"] = (
            df["ship_type"].map(SHIP_TYPE_MAP).fillna(0).astype("int16")
        )
        return df

    # ?? 8. Dimension imputation from antenna offsets ?????????????????????????
    def impute_dimensions_from_antenna(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        AIS Message Type 5 encodes antenna reference position offsets:
          A = bow to antenna,  B = stern to antenna  ->Length = A + B
          C = port to antenna, D = starboard to antenna ->Width = C + D
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

    # ?? 9. Missing value imputation ?????????????????????????????????????????
    def impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Per-column strategy:
          vessel_name / imo / callsign : fillna("UNKNOWN") already from loader
          ship_type_code               : per-MMSI mode ->0 (Unknown)
          length / width / draught     : per-MMSI median ->per-ship_type median
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

        # Dimensions: per-MMSI median ->per ship_type median
        for col in ["length", "width", "draught"]:
            # Handle empty groups by using transform with proper NaN handling
            df[col] = df.groupby("mmsi")[col].transform(
                lambda x: x.fillna(x.median()) if not x.isna().all() else x
            )
            # Fill remaining NaN with ship_type median
            ship_median = df.groupby("ship_type_code")[col].transform("median")
            df[col] = df[col].fillna(ship_median)

        # Parse ETA as datetime (reuse timestamp format)
        df["eta_dt"] = pd.to_datetime(
            df["eta"], format="%d/%m/%Y %H:%M:%S", utc=True, errors="coerce"
        )
        return df

    # ?? 10. Deduplication ???????????????????????????????????????????????????
    def deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        n = len(df)
        df = df.drop_duplicates(subset=["mmsi", "timestamp"], keep="first")
        self.report["duplicates_removed"] = n - len(df)
        logger.info("Deduplication: removed %d records", n - len(df))
        return df

    # ?? Pipeline runner ?????????????????????????????????????????????????????
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Input: %d records", len(df))
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
        logger.info("Clean records: %d -> %s", len(df), self.output_path)
        logger.info("Report: %s", self.report)
        return df


def main():
    parser = argparse.ArgumentParser(description="Clean aisdk data")
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument("--output", default="outputs/processed/ais_clean.parquet",
                        help="Output parquet file")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    df = pd.read_parquet(args.input)
    logger.info("Loaded %d records from %s", len(df), args.input)

    cleaner = AISCleaner(output_path=args.output)
    df_clean = cleaner.run(df)

    print(f"Cleaned {len(df_clean):,} records -> {args.output}")
    print(f"Cleaning report: {cleaner.report}")


if __name__ == "__main__":
    main()

