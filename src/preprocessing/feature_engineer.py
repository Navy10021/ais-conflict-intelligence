"""
AIS Feature Engineer (aisdk-specific)
=====================================
Six feature categories:

  A. Kinematic Features       ??speed, course, heading, maneuver dynamics
  B. Rate-of-Turn Features    ??ROT-based maneuver classification (NEW)
  C. Geospatial Features      ??conflict zone membership, chokepoint distance
  D. Behavioral Features      ??rolling window irregularity metrics
  E. Destination/ETA Features ??port call analysis & ETA deviation (NEW)
  F. Temporal Aggregation     ??grid 횞 6h bucket statistics
  G. Conflict Labels          ??binary label + regression target
"""
import pandas as pd
import numpy as np
from scipy import stats
import logging
import argparse
from pathlib import Path
from typing import Callable, Any
import yaml

logger = logging.getLogger(__name__)


class AISFeatureEngineer:

    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config = self._load_config(config_path)
        self.required = self.config.get("validation", {}).get("feature_required_columns", {})

    def _load_config(self, path: str) -> dict:
        p = Path(path)
        if not p.exists():
            return {}
        with p.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _require_columns(self, df: pd.DataFrame, key: str) -> None:
        cols = set(self.required.get(key, []))
        missing = sorted(cols - set(df.columns))
        if missing:
            raise ValueError(f"[{key}] Missing required columns: {missing}")

    CONFLICT_ZONES = {
        "black_sea":     {"bbox": [27.0, 40.5, 41.0, 46.8], "conflict": "ukraine_war"},
        "azov_sea":      {"bbox": [33.5, 45.0, 39.5, 47.5], "conflict": "ukraine_war"},
        "kerch_strait":  {"bbox": [36.4, 45.1, 36.8, 45.5], "conflict": "ukraine_war"},
        "red_sea":       {"bbox": [32.0, 12.0, 43.5, 30.0], "conflict": "houthi_crisis"},
        "bab_el_mandeb": {"bbox": [43.0, 11.5, 45.0, 12.5], "conflict": "houthi_crisis"},
        "taiwan_strait": {"bbox": [119.0, 22.0, 122.0, 26.0], "conflict": "taiwan_tension"},
        "south_china_sea": {"bbox": [109.0, 3.0, 121.0, 22.0], "conflict": "scs_dispute"},
        "strait_hormuz": {"bbox": [56.0, 25.5, 59.5, 27.0], "conflict": "iran_tension"},
        "north_sea":     {"bbox": [-5.0, 51.0, 10.0, 58.0], "conflict": "none"},
        "baltic_sea":    {"bbox": [10.0, 54.0, 30.0, 65.0], "conflict": "none"},
    }

    CHOKEPOINTS = {
        "hormuz":     (56.50, 26.50),
        "malacca":    (103.80, 1.20),
        "bab_mandeb": (43.40, 12.50),
        "suez":       (32.50, 30.70),
        "panama":     (-79.90, 9.00),
        "gibraltar":  (-5.40, 36.00),
        "dover":      (1.30, 51.00),
        "danish_straits": (12.60, 55.70),
        "skagerrak":  (9.50, 57.80),
    }

    # ?? A. Kinematic Features ???????????????????????????????????????????????
    def add_kinematic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self._require_columns(df, "kinematic")
        """
        speed_category    : anchored / drifting / slow / cruising / fast
        delta_sog         : |SOG_t ??SOG_{t??}| per MMSI
        delta_cog         : COG change, 360째 wrap-corrected
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

    # ?? B. Rate-of-Turn Features (NEW ??aisdk has ROT column) ???????????????
    def add_rot_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self._require_columns(df, "rot")
        """
        ROT (deg/min) is available in aisdk ??critical for maneuver classification.

        rot_category : "hard_port" | "port" | "steady" | "starboard" | "hard_starboard"
        rot_abs     : |ROT|
        rot_spike   : ROT changes >20 deg/min between consecutive fixes
        evasive_maneuver : high ROT (>10 deg/min) + high SOG (>5 kn) in conflict zone
        rot_vs_cog_consistency : |rot| > 5 but delta_cog < 1째 ??sensor anomaly flag
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

    # ?? C. Geospatial Features ???????????????????????????????????????????????
    def add_geospatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self._require_columns(df, "geospatial")
        """
        grid_cell         : 0.5째 횞 0.5째 cell ID
        in_conflict_zone  : bool ??inside any conflict zone
        conflict_zone_name: zone label or "none"
        is_control_zone   : bool ??inside a designated control zone
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
                df.loc[mask, "in_conflict_zone"] = True
                df.loc[mask, "conflict_zone_name"] = zone

        for name, (cp_lon, cp_lat) in self.CHOKEPOINTS.items():
            col_name = f"dist_{name}_km"
            df[col_name] = self._haversine(df["lat"], df["lon"], cp_lat, cp_lon)

        return df

    # ?? D. Behavioral Features ????????????????????????????????????????????????
    def add_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self._require_columns(df, "behavioral")
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
            roll = grp[col].rolling("12h", min_periods=3)
            df[f"rolling_{label}_mean_12h"] = roll.mean().reset_index(level=0, drop=True)
            df[f"rolling_{label}_std_12h"]  = roll.std().reset_index(level=0, drop=True)

        df = df.reset_index()

        def _entropy(s: pd.Series) -> float:
            s_clean = s.dropna()
            if len(s_clean) < 10:
                return 0.0
            try:
                bins = pd.cut(s_clean, bins=36)
                p = bins.value_counts(normalize=True) + 1e-10
                return float(stats.entropy(p))
            except Exception:
                return 0.0

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

    # ?? E. Destination / ETA Features (NEW ??aisdk has Destination & ETA) ???
    def add_destination_features(
        self, df: pd.DataFrame, ports_path: str
    ) -> pd.DataFrame:
        self._require_columns(df, "destination")
        """
        Exploit the `destination` and `eta_dt` columns unique to aisdk.

        dest_is_conflict_port     : declared destination is a port in a conflict zone
        dest_changed              : vessel changed destination between consecutive fixes
        eta_hours_remaining       : hours until ETA from current fix
        eta_implausible           : ETA is in the past (spoofing / stale data)
        destination_count_24h     : unique destinations declared in 24h (high = suspicious)
        """
        try:
            ports = pd.read_csv(ports_path)
            required_port_cols = {"conflict_zone", "locode"}
            missing_port_cols = required_port_cols - set(ports.columns)
            if missing_port_cols:
                logger.warning("Ports schema missing columns %s. Skipping destination features.", sorted(missing_port_cols))
                return df
            conflict_ports = set(ports.loc[ports["conflict_zone"] != "none", "locode"].astype(str).str.upper())
        except FileNotFoundError:
            logger.warning("Ports file not found: {ports_path}. Skipping destination features.")
            return df

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

    # ?? F. Temporal Aggregation ???????????????????????????????????????????????
    def add_temporal_aggregation(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Aggregate AIS data by grid cell and 6-hour time bucket.
        """
        df = df.copy()

        if "grid_cell" not in df.columns:
            logger.info("Creating grid cells...")
            df["grid_lat"] = (df["lat"] // 0.5) * 0.5
            df["grid_lon"] = (df["lon"] // 0.5) * 0.5
            df["grid_cell"] = df["grid_lat"].astype(str) + "_" + df["grid_lon"].astype(str)

        if "time_bucket" not in df.columns:
            logger.info("Creating time buckets...")
            df["time_bucket"] = df["timestamp"].dt.floor("6h")

        logger.info("Aggregating by grid_cell and time_bucket...")

        # Build aggregation dict dynamically based on available columns
        agg_dict: dict[str, tuple[str, str] | tuple[str, Callable[[Any], Any]]] = {
            "traffic_count": ("mmsi", "nunique"),
        }

        if "mobile_type" in df.columns:
            agg_dict["class_a_count"] = ("mobile_type", lambda x: (x == "Class A").sum())
        if "is_dark_ship" in df.columns:
            agg_dict["dark_ship_count"] = ("is_dark_ship", "sum")
        if "ship_type_code" in df.columns:
            agg_dict["military_count"] = ("ship_type_code", lambda x: (x == 35).sum())
            agg_dict["fishing_count"] = ("ship_type_code", lambda x: (x == 30).sum())
            agg_dict["tanker_count"] = ("ship_type_code", lambda x: x.isin(range(80, 90)).sum())
            agg_dict["cargo_count"] = ("ship_type_code", lambda x: x.isin(range(70, 80)).sum())
            agg_dict["sar_count"] = ("ship_type_code", lambda x: (x == 51).sum())
        if "sog" in df.columns:
            agg_dict["mean_sog"] = ("sog", "mean")
            agg_dict["std_sog"] = ("sog", "std")
        if "rot_abs" in df.columns:
            agg_dict["mean_rot_abs"] = ("rot_abs", "mean")
        if "loitering_flag" in df.columns:
            agg_dict["loitering_density"] = ("loitering_flag", "sum")
        if "not_under_command" in df.columns:
            agg_dict["not_under_command_count"] = ("not_under_command", "sum")
        if "evasive_maneuver" in df.columns:
            agg_dict["evasive_count"] = ("evasive_maneuver", "sum")
        if "dest_is_conflict_port" in df.columns:
            agg_dict["dest_conflict_count"] = ("dest_is_conflict_port", "sum")

        agg = df.groupby(["grid_cell", "time_bucket"]).agg(**agg_dict).reset_index()

        # Compute ratios only if we have traffic_count and the relevant count columns
        if "traffic_count" in agg.columns:
            denom = agg["traffic_count"].clip(lower=1)
            if "class_a_count" in agg.columns:
                agg["class_a_ratio"] = agg["class_a_count"] / denom
            if "dark_ship_count" in agg.columns:
                agg["dark_ship_ratio"] = agg["dark_ship_count"] / denom
            if "military_count" in agg.columns:
                agg["military_ratio"] = agg["military_count"] / denom
            if "fishing_count" in agg.columns:
                agg["fishing_ratio"] = agg["fishing_count"] / denom
            if "tanker_count" in agg.columns:
                agg["tanker_ratio"] = agg["tanker_count"] / denom

        self.agg_df = agg
        return df, agg

    # ?? G. Conflict Labels ????????????????????????????????????????????????????
    def add_conflict_labels(
        self, df: pd.DataFrame, conflict_events_path: str
    ) -> pd.DataFrame:
        """
        conflict_label     : 1 if conflict event occurs within 30 days in same zone
        days_to_conflict   : signed days to nearest event (regression target)
        conflict_intensity : ACLED fatality count (continuous severity proxy)
        """
        if not conflict_events_path:
            logger.warning("No conflict events path provided. Skipping labels.")
            return df
        try:
            events = pd.read_csv(conflict_events_path, parse_dates=["event_date"])
        except FileNotFoundError:
            logger.warning("Conflict events file not found: %s. Skipping labels.", conflict_events_path)
            return df

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

    # ?? util ??????????????????????????????????????????????????????????????????
    @staticmethod
    def _haversine(lat1, lon1, lat2: float, lon2: float) -> pd.Series:
        R = 6_371.0
        p1 = np.radians(lat1)
        p2 = np.radians(lat2)
        dp = np.radians(lat2 - lat1)
        dl = np.radians(lon2 - lon1)
        a = np.sin(dp / 2)**2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2)**2
        return R * 2 * np.arcsin(np.sqrt(a))

    def run(
        self, df: pd.DataFrame,
        conflict_events_path: str = "",
        ports_path: str = "",
    ) -> pd.DataFrame:
        logger.info("A. Kinematic features...")
        df = self.add_kinematic_features(df)
        logger.info("C. Geospatial features...")
        df = self.add_geospatial_features(df)
        logger.info("B. ROT features...")
        df = self.add_rot_features(df)
        logger.info("D. Behavioral features...")
        df = self.add_behavioral_features(df)
        logger.info("E. Destination/ETA features...")
        if ports_path:
            df = self.add_destination_features(df, ports_path)
        else:
            logger.warning("No ports path provided. Skipping destination features.")
        logger.info("F. Temporal aggregation...")
        df, _ = self.add_temporal_aggregation(df)
        logger.info("G. Conflict labels...")
        df = self.add_conflict_labels(df, conflict_events_path)
        return df


def main():
    parser = argparse.ArgumentParser(description="Generate AIS features")
    parser.add_argument("--input", required=True, help="Input parquet file (cleaned)")
    parser.add_argument("--output", default="outputs/processed/ais_features.parquet",
                        help="Output parquet file")
    parser.add_argument("--conflict-events", default="",
                        help="Path to ACLED events CSV")
    parser.add_argument("--ports", default="",
                        help="Path to world_ports.csv")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    df = pd.read_parquet(args.input)
    logger.info("Loaded {len(df):,} records from {args.input}")

    engineer = AISFeatureEngineer()
    df_features = engineer.run(
        df,
        conflict_events_path=args.conflict_events,
        ports_path=args.ports,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_parquet(output_path, index=False, compression="snappy")
    logger.info("Saved {len(df_features):,} records to {output_path}")
    print(f"Feature engineering complete: {len(df_features):,} records ??{args.output}")


if __name__ == "__main__":
    main()

