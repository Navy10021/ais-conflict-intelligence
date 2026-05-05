"""
AIS Data Aggregator
===================
Aggregates cleaned AIS data into grid_cell 횞 time_bucket statistics.
"""
import pandas as pd
from pathlib import Path
import logging
import argparse
from typing import Callable, Any

logger = logging.getLogger(__name__)


class AISAggregator:
    """
    Aggregates AIS data by grid cell and time bucket.

    Grid: 0.5째 횞 0.5째 cells
    Time bucket: 6-hour intervals
    """

    def __init__(self, output_path: str = "outputs/processed/ais_aggregated.parquet"):
        self.output_path = Path(output_path)

    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate AIS data by grid_cell and 6-hour time bucket.

        Returns:
            DataFrame with columns:
            - grid_cell, time_bucket
            - traffic_count, class_a_ratio, dark_ship_ratio
            - military_ratio, tanker_ratio, sar_count
            - mean_sog, std_sog, mean_rot_abs
            - loitering_density, not_under_command_count
            - evasive_count, dest_conflict_count
        """
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

        logger.info("Computing ratios...")
        denom = agg["traffic_count"].clip(lower=1)
        if "class_a_count" in agg.columns:
            agg["class_a_ratio"] = agg["class_a_count"] / denom
        if "dark_ship_count" in agg.columns:
            agg["dark_ship_ratio"]  = agg["dark_ship_count"] / denom
        if "military_count" in agg.columns:
            agg["military_ratio"]   = agg["military_count"]   / denom
        if "fishing_count" in agg.columns:
            agg["fishing_ratio"]    = agg["fishing_count"]    / denom
        if "tanker_count" in agg.columns:
            agg["tanker_ratio"]     = agg["tanker_count"]     / denom

        # Drop intermediate count columns
        drop_cols = [
            "class_a_count", "dark_ship_count", "military_count",
            "fishing_count", "tanker_count", "cargo_count"
        ]
        agg = agg.drop(columns=[c for c in drop_cols if c in agg.columns])

        logger.info("Aggregation complete: {len(agg):,} rows")
        return agg

    def run(self, input_path: str) -> pd.DataFrame:
        """Load parquet, aggregate, and save."""
        logger.info("Loading {input_path}...")
        df = pd.read_parquet(input_path)

        agg_df = self.aggregate(df)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        agg_df.to_parquet(self.output_path, index=False, compression="snappy")
        logger.info("Saved aggregated data to {self.output_path}")

        return agg_df


def main():
    parser = argparse.ArgumentParser(description="Aggregate AIS data by grid and time")
    parser.add_argument("--input", required=True, help="Input parquet file (features)")
    parser.add_argument("--output", default="outputs/processed/ais_aggregated.parquet",
                        help="Output parquet file")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    aggregator = AISAggregator(output_path=args.output)
    agg_df = aggregator.run(args.input)

    print(f"Aggregation complete: {len(agg_df):,} rows ??{args.output}")


if __name__ == "__main__":
    main()

