"""
Rate-of-Turn (ROT) Analysis
==========================
Analyze ROT patterns, maneuvers, and anomalies.
"""
import pandas as pd
from pathlib import Path
import logging
import argparse

logger = logging.getLogger(__name__)


class ROTAnalyzer:
    """Analyze Rate-of-Turn patterns and anomalies."""

    def __init__(self, output_dir: str = "outputs/tables"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def rot_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute ROT distribution statistics by zone."""
        logger.info("Computing ROT distribution...")
        if "rot" not in df.columns:
            logger.warning("rot column not found. Skipping.")
            return pd.DataFrame()

        df = df.dropna(subset=["rot"])
        stats = (
            df.groupby("conflict_zone_name")["rot"]
            .agg(["mean", "std", "min", "max", "count"])
            .reset_index()
        )

        output_path = self.output_dir / "rot_distribution.csv"
        stats.to_csv(output_path, index=False)
        logger.info("Saved to {output_path}")
        return stats

    def rot_spike_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze ROT spike events."""
        logger.info("Analyzing ROT spikes...")
        if "rot_spike" not in df.columns:
            logger.warning("rot_spike column not found. Skipping.")
            return pd.DataFrame()

        spikes = (
            df.groupby(["conflict_zone_name", "in_conflict_zone"])["rot_spike"]
            .agg(["sum", "mean", "count"])
            .reset_index()
        )

        output_path = self.output_dir / "rot_spikes.csv"
        spikes.to_csv(output_path, index=False)
        logger.info("Saved to {output_path}")
        return spikes

    def evasive_maneuver_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze evasive maneuver patterns."""
        logger.info("Computing evasive maneuver statistics...")
        if "evasive_maneuver" not in df.columns:
            logger.warning("evasive_maneuver column not found. Skipping.")
            return pd.DataFrame()

        evasive = (
            df.groupby(["conflict_zone_name", "in_conflict_zone"])["evasive_maneuver"]
            .agg(["sum", "mean", "count"])
            .reset_index()
        )

        output_path = self.output_dir / "evasive_maneuvers.csv"
        evasive.to_csv(output_path, index=False)
        logger.info("Saved to {output_path}")
        return evasive

    def rot_cog_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze ROT vs COG consistency issues."""
        logger.info("Analyzing ROT-COG consistency...")
        if "rot_cog_inconsistent" not in df.columns:
            logger.warning("rot_cog_inconsistent column not found. Skipping.")
            return pd.DataFrame()

        consistency = (
            df.groupby("conflict_zone_name")["rot_cog_inconsistent"]
            .agg(["sum", "mean", "count"])
            .reset_index()
        )

        output_path = self.output_dir / "rot_cog_consistency.csv"
        consistency.to_csv(output_path, index=False)
        logger.info("Saved to {output_path}")
        return consistency


def main():
    parser = argparse.ArgumentParser(description="ROT analysis")
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

    analyzer = ROTAnalyzer(output_dir=args.output)
    analyzer.rot_distribution(df)
    analyzer.rot_spike_analysis(df)
    analyzer.evasive_maneuver_stats(df)
    analyzer.rot_cog_consistency(df)

    print(f"ROT analysis saved to {args.output}")


if __name__ == "__main__":
    main()

