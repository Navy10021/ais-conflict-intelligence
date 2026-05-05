"""
Traffic Volume & Density Analysis
================================
Analyze vessel traffic patterns, volume changes, and density metrics.
"""
import pandas as pd
from pathlib import Path
import logging
import argparse

logger = logging.getLogger(__name__)


class TrafficAnalyzer:
    """Analyze traffic volume and density patterns."""

    def __init__(self, output_dir: str = "outputs/tables"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def daily_traffic_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute daily unique vessel count and total records."""
        logger.info("Computing daily traffic volume...")
        df = df.copy()
        df["date"] = df["timestamp"].dt.date

        daily = (
            df.groupby("date")
            .agg(
                unique_vessels=("mmsi", "nunique"),
                total_records=("mmsi", "count"),
                avg_sog=("sog", "mean"),
            )
            .reset_index()
        )

        output_path = self.output_dir / "daily_traffic_volume.csv"
        daily.to_csv(output_path, index=False)
        logger.info("Saved to {output_path}")
        return daily

    def traffic_by_zone(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute traffic metrics by conflict zone."""
        logger.info("Computing traffic by zone...")
        df = df.copy()
        df["date"] = df["timestamp"].dt.date

        zone_traffic = (
            df.groupby(["date", "conflict_zone_name"])
            .agg(
                unique_vessels=("mmsi", "nunique"),
                military_count=("ship_type_code", lambda x: (x == 35).sum()),
                cargo_count=("ship_type_code", lambda x: x.isin(range(70, 80)).sum()),
                tanker_count=("ship_type_code", lambda x: x.isin(range(80, 90)).sum()),
            )
            .reset_index()
        )

        output_path = self.output_dir / "traffic_by_zone.csv"
        zone_traffic.to_csv(output_path, index=False)
        logger.info("Saved to {output_path}")
        return zone_traffic

    def density_by_grid(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute vessel density by grid cell."""
        logger.info("Computing density by grid cell...")
        df = df.copy()

        if "grid_cell" not in df.columns:
            df["grid_lat"] = (df["lat"] // 0.5) * 0.5
            df["grid_lon"] = (df["lon"] // 0.5) * 0.5
            df["grid_cell"] = df["grid_lat"].astype(str) + "_" + df["grid_lon"].astype(str)

        density = (
            df.groupby("grid_cell")
            .agg(
                unique_vessels=("mmsi", "nunique"),
                avg_lat=("lat", "mean"),
                avg_lon=("lon", "mean"),
                in_conflict_zone=("in_conflict_zone", "first"),
            )
            .reset_index()
        )

        output_path = self.output_dir / "grid_density.csv"
        density.to_csv(output_path, index=False)
        logger.info("Saved to {output_path}")
        return density


def main():
    parser = argparse.ArgumentParser(description="Traffic analysis")
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument("--output", default="outputs/tables")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    df = pd.read_parquet(args.input)
    logger.info("Loaded {len(df):,} records")

    analyzer = TrafficAnalyzer(output_dir=args.output)
    analyzer.daily_traffic_volume(df)
    analyzer.traffic_by_zone(df)
    analyzer.density_by_grid(df)

    print(f"Traffic analysis saved to {args.output}")


if __name__ == "__main__":
    main()

