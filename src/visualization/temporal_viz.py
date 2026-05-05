"""
Temporal Visualization Module
=======================
Outputs ??outputs/figures/temporal/

Figures:
  1. Daily traffic volume per conflict zone ??line chart + conflict onset axvlines
  2. Vessel type composition ??stacked area chart
  3. Speed distribution change ??violin plots per zone
  4. Dark ship ratio time series ??7-day rolling average
  5. Mean |ROT| time series ??maneuver intensity over time
  6. Destination diversity index ??unique destinations per zone per day
  7. CCF: AIS indicators vs. conflict intensity ??짹30-day lead/lag
"""
import pandas as pd
from pathlib import Path
import logging
import argparse
from typing import Optional

logger = logging.getLogger(__name__)


class TemporalVisualizer:
    """Generate temporal visualizations for AIS data."""

    def __init__(self, output_dir: str = "outputs/figures/temporal"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_plot_style()

    def _setup_plot_style(self):
        """Set up matplotlib for publication-quality figures."""
        try:
            import matplotlib.pyplot as plt
            plt.rcParams.update({
                "figure.dpi": 300,
                "figure.figsize": (10, 6),
                "font.family": "DejaVu Sans",
                "font.size": 12,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.grid": True,
                "grid.alpha": 0.3,
            })
        except ImportError:
            logger.warning("matplotlib not installed.")

    def plot_daily_traffic(
        self, df: pd.DataFrame, conflict_onsets: Optional[dict] = None
    ) -> str:
        """
        Plot daily traffic volume per conflict zone.
        """
        try:
            import matplotlib.pyplot as plt  # noqa: F401
        except ImportError:
            logger.error("matplotlib not installed.")
            return ""

        logger.info("Plotting daily traffic volume...")
        df = df.dropna(subset=["timestamp", "in_conflict_zone"])
        df["date"] = df["timestamp"].dt.date

        daily = (
            df.groupby(["date", "in_conflict_zone"])["mmsi"]
            .nunique()
            .reset_index()
        )

        plt.figure(figsize=(12, 6))
        for zone, group in daily.groupby("in_conflict_zone"):
            plt.plot(group["date"], group["mmsi"], label=f"Conflict Zone: {zone}")

        if conflict_onsets:
            for onset in conflict_onsets.values():
                plt.axvline(pd.Timestamp(onset), color="red", linestyle="--", alpha=0.7)

        plt.xlabel("Date")
        plt.ylabel("Unique Vessels")
        plt.title("Daily Traffic Volume")
        plt.legend()
        plt.tight_layout()

        output_path = self.output_dir / "daily_traffic_volume.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved to {output_path}")
        return str(output_path)

    def plot_vessel_composition(self, df: pd.DataFrame) -> str:
        """
        Stacked area chart of vessel type composition.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib not installed.")
            return ""

        logger.info("Plotting vessel type composition...")
        df = df.dropna(subset=["timestamp", "ship_type_code"])
        df["date"] = df["timestamp"].dt.date

        type_map = {30: "Fishing", 35: "Military", 51: "SAR", 70: "Cargo", 80: "Tanker"}
        df["type_label"] = df["ship_type_code"].map(lambda x: type_map.get(x, "Other"))

        daily = (
            df.groupby(["date", "type_label"])["mmsi"]
            .nunique()
            .reset_index()
        )
        pivot = daily.pivot(index="date", columns="type_label", values="mmsi").fillna(0)

        plt.figure(figsize=(12, 6))
        plt.stackplot(pivot.index, pivot.T.values, labels=pivot.columns, alpha=0.8)
        plt.xlabel("Date")
        plt.ylabel("Unique Vessels")
        plt.title("Vessel Type Composition Over Time")
        plt.legend(loc="upper right")
        plt.tight_layout()

        output_path = self.output_dir / "vessel_composition.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved to {output_path}")
        return str(output_path)

    def plot_speed_distribution(self, df: pd.DataFrame) -> str:
        """
        Violin plots of speed distribution per zone.
        """
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("seaborn/matplotlib not installed.")
            return ""

        logger.info("Plotting speed distribution...")
        df = df.dropna(subset=["sog", "in_conflict_zone"])

        plt.figure(figsize=(10, 6))
        sns.violinplot(data=df, x="in_conflict_zone", y="sog", inner="quartile")
        plt.xlabel("Conflict Zone")
        plt.ylabel("Speed Over Ground (knots)")
        plt.title("Speed Distribution by Zone")
        plt.tight_layout()

        output_path = self.output_dir / "speed_distribution.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved to {output_path}")
        return str(output_path)

    def plot_dark_ship_ratio(self, df: pd.DataFrame) -> str:
        """
        7-day rolling average of dark ship ratio.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib not installed.")
            return ""

        logger.info("Plotting dark ship ratio...")
        df = df.dropna(subset=["timestamp", "is_dark_ship"]).copy()
        df["date"] = df["timestamp"].dt.date

        daily = (
            df.groupby("date")["is_dark_ship"]
            .mean()
            .reset_index()
        )
        daily["rolling_7d"] = daily["is_dark_ship"].rolling(7, min_periods=1).mean()

        plt.figure(figsize=(12, 6))
        plt.plot(daily["date"], daily["rolling_7d"], color="red", linewidth=2)
        plt.fill_between(daily["date"], daily["rolling_7d"], alpha=0.3, color="red")
        plt.xlabel("Date")
        plt.ylabel("Dark Ship Ratio (7-day avg)")
        plt.title("Dark Ship Ratio Over Time")
        plt.tight_layout()

        output_path = self.output_dir / "dark_ship_ratio.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved to {output_path}")
        return str(output_path)

    def plot_mean_rot(self, df: pd.DataFrame) -> str:
        """
        Time series of mean |ROT| - maneuver intensity.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib not installed.")
            return ""

        logger.info("Plotting mean |ROT| time series...")
        if "rot_abs" not in df.columns:
            logger.warning("rot_abs column not found. Skipping.")
            return ""

        df = df.dropna(subset=["timestamp", "rot_abs"]).copy()
        df["date"] = df["timestamp"].dt.date

        daily = (
            df.groupby("date")["rot_abs"]
            .mean()
            .reset_index()
        )

        plt.figure(figsize=(12, 6))
        plt.plot(daily["date"], daily["rot_abs"], color="purple", linewidth=2)
        plt.xlabel("Date")
        plt.ylabel("Mean |ROT| (deg/min)")
        plt.title("Maneuver Intensity (Mean |ROT|) Over Time")
        plt.tight_layout()

        output_path = self.output_dir / "mean_rot_timeseries.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved to {output_path}")
        return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate temporal visualizations")
    parser.add_argument("--input", required=True, help="Input parquet file (features)")
    parser.add_argument("--output-dir", default="outputs/figures/temporal",
                        help="Output directory for figures")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    df = pd.read_parquet(args.input)
    logger.info("Loaded {len(df):,} records from {args.input}")

    viz = TemporalVisualizer(output_dir=args.output_dir)

    viz.plot_daily_traffic(df)
    viz.plot_vessel_composition(df)
    viz.plot_speed_distribution(df)
    viz.plot_dark_ship_ratio(df)
    viz.plot_mean_rot(df)

    print(f"Temporal visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()

