"""
Destination & Port Call Analysis
==============================
Analyze port calls, destination changes, and ETA deviations.
"""
import pandas as pd
from pathlib import Path
import logging
import argparse

logger = logging.getLogger(__name__)


class DestinationAnalyzer:
    """Analyze vessel destination patterns."""

    def __init__(self, output_dir: str = "outputs/tables"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def port_call_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute port call frequency by destination."""
        logger.info("Computing port call frequency...")
        df = df.dropna(subset=["destination"])
        df = df[df["destination"] != "UNKNOWN"]

        freq = (
            df.groupby("destination")
            .agg(
                call_count=("mmsi", "count"),
                unique_vessels=("mmsi", "nunique"),
                avg_sog=("sog", "mean"),
            )
            .sort_values("call_count", ascending=False)
            .head(50)
            .reset_index()
        )

        output_path = self.output_dir / "port_call_frequency.csv"
        freq.to_csv(output_path, index=False)
        logger.info("Saved to {output_path}")
        return freq

    def destination_change_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze destination change patterns."""
        logger.info("Computing destination change rate...")
        if "dest_changed" not in df.columns:
            logger.warning("dest_changed column not found. Skipping.")
            return pd.DataFrame()

        df = df.copy()
        df["date"] = df["timestamp"].dt.date

        change_rate = (
            df.groupby("date")["dest_changed"]
            .mean()
            .reset_index()
        )
        change_rate.columns = ["date", "change_rate"]

        output_path = self.output_dir / "destination_change_rate.csv"
        change_rate.to_csv(output_path, index=False)
        logger.info("Saved to {output_path}")
        return change_rate

    def eta_plausibility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze ETA plausibility and spoofs."""
        logger.info("Analyzing ETA plausibility...")
        if "eta_implausible" not in df.columns:
            logger.warning("eta_implausible column not found. Skipping.")
            return pd.DataFrame()

        eta_stats = (
            df.groupby(["conflict_zone_name", "in_conflict_zone"])["eta_implausible"]
            .agg(["sum", "mean", "count"])
            .reset_index()
        )

        output_path = self.output_dir / "eta_plausibility.csv"
        eta_stats.to_csv(output_path, index=False)
        logger.info("Saved to {output_path}")
        return eta_stats

    def conflict_port_destinations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze vessels declaring conflict zone ports as destination."""
        logger.info("Analyzing conflict port destinations...")
        if "dest_is_conflict_port" not in df.columns:
            logger.warning("dest_is_conflict_port column not found. Skipping.")
            return pd.DataFrame()

        conflict_dest = (
            df.groupby("date")["dest_is_conflict_port"]
            .agg(["sum", "mean", "count"])
            .reset_index()
        )
        conflict_dest.columns = ["date", "conflict_dest_count", "conflict_dest_rate", "total"]

        output_path = self.output_dir / "conflict_port_destinations.csv"
        conflict_dest.to_csv(output_path, index=False)
        logger.info("Saved to {output_path}")
        return conflict_dest


def main():
    parser = argparse.ArgumentParser(description="Destination analysis")
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

    analyzer = DestinationAnalyzer(output_dir=args.output)
    analyzer.port_call_frequency(df)
    analyzer.destination_change_rate(df)
    analyzer.eta_plausibility(df)
    analyzer.conflict_port_destinations(df)

    print(f"Destination analysis saved to {args.output}")


if __name__ == "__main__":
    main()

