"""
Behavioral Pattern Analysis
=======================
Analyze vessel behavior patterns and anomalies.
"""
import pandas as pd
from pathlib import Path
import logging
import argparse

logger = logging.getLogger(__name__)


class BehavioralAnalyzer:
    """Analyze vessel behavioral patterns."""

    def __init__(self, output_dir: str = "outputs/tables"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def route_entropy_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze route entropy by vessel and zone."""
        logger.info("Computing route entropy statistics...")
        df = df.copy()

        if "route_entropy" not in df.columns:
            logger.warning("route_entropy not found. Skipping.")
            return pd.DataFrame()

        stats = (
            df.groupby("conflict_zone_name")["route_entropy"]
            .agg(["mean", "std", "median", "count"])
            .reset_index()
        )

        output_path = self.output_dir / "route_entropy_stats.csv"
        stats.to_csv(output_path, index=False)
        logger.info("Saved to {output_path}")
        return stats

    def loitering_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze loitering behavior in conflict zones."""
        logger.info("Computing loitering statistics...")
        df = df.copy()

        if "loitering_flag" not in df.columns:
            logger.warning("loitering_flag not found. Skipping.")
            return pd.DataFrame()

        loitering = (
            df.groupby(["conflict_zone_name", "in_conflict_zone"])["loitering_flag"]
            .agg(["sum", "mean", "count"])
            .reset_index()
        )

        output_path = self.output_dir / "loitering_stats.csv"
        loitering.to_csv(output_path, index=False)
        logger.info("Saved to {output_path}")
        return loitering

    def zigzag_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze zig-zag index patterns."""
        logger.info("Computing zig-zag statistics...")
        df = df.copy()

        if "zig_zag_index" not in df.columns:
            logger.warning("zig_zag_index not found. Skipping.")
            return pd.DataFrame()

        zigzag = (
            df.groupby("conflict_zone_name")["zig_zag_index"]
            .agg(["mean", "std", "max", "count"])
            .reset_index()
        )

        output_path = self.output_dir / "zigzag_stats.csv"
        zigzag.to_csv(output_path, index=False)
        logger.info("Saved to {output_path}")
        return zigzag


def main():
    parser = argparse.ArgumentParser(description="Behavioral analysis")
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

    analyzer = BehavioralAnalyzer(output_dir=args.output)
    analyzer.route_entropy_analysis(df)
    analyzer.loitering_analysis(df)
    analyzer.zigzag_analysis(df)

    print(f"Behavioral analysis saved to {args.output}")


if __name__ == "__main__":
    main()

