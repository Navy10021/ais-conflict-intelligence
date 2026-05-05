"""
Statistical Visualization Module
===========================
Outputs ??outputs/figures/statistical/

Figures:
  1. Pre/post conflict distributions ??KDE + boxplot for each feature
  2. Spearman correlation heatmap ??AIS features 횞 conflict intensity
  3. Feature importance bar chart ??Random Forest / XGBoost
  4. ROC-AUC comparison ??LSTM / TFT / XGBoost / Prophet baseline
  5. Precision?밨ecall curves ??imbalanced label evaluation
  6. SHAP summary ??top-20 features, bee-swarm plot
  7. Confusion matrices ??T+7, T+14, T+30 horizons per zone
  8. ROT distribution pre/post ??violin plots
  9. Destination change rate ??pre/post conflict comparison
"""
import pandas as pd
from pathlib import Path
import logging
import argparse
from typing import Optional

logger = logging.getLogger(__name__)


class StatisticalVisualizer:
    """Generate statistical visualizations for AIS data."""

    def __init__(self, output_dir: str = "outputs/figures/statistical"):
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

    def plot_distribution_comparison(
        self, df: pd.DataFrame, feature: str, group_col: str = "conflict_label"
    ) -> str:
        """
        KDE + boxplot for feature distribution by group.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.error("matplotlib/seaborn not installed.")
            return ""

        logger.info("Plotting distribution comparison for {feature}...")
        df = df.dropna(subset=[feature, group_col])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # KDE plot
        for group_val in df[group_col].unique():
            subset = df[df[group_col] == group_val][feature]
            sns.kdeplot(subset, ax=axes[0], label=f"Group {group_val}", fill=True, alpha=0.5)
        axes[0].set_title(f"KDE: {feature}")
        axes[0].legend()

        # Boxplot
        sns.boxplot(data=df, x=group_col, y=feature, ax=axes[1])
        axes[1].set_title(f"Boxplot: {feature}")

        plt.tight_layout()
        output_path = self.output_dir / f"dist_{feature}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved to {output_path}")
        return str(output_path)

    def plot_correlation_heatmap(
        self, df: pd.DataFrame, features: Optional[list] = None
    ) -> str:
        """
        Spearman correlation heatmap of AIS features.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.error("matplotlib/seaborn not installed.")
            return ""

        logger.info("Plotting correlation heatmap...")
        if features is None:
            features = [
                "sog", "cog", "rot", "rot_abs", "delta_sog", "delta_cog",
                "is_dark_ship", "sog_z_score", "route_entropy", "zig_zag_index",
            ]

        available = [f for f in features if f in df.columns]
        if len(available) < 2:
            logger.warning("Not enough features for correlation.")
            return ""

        corr = df[available].corr(method="spearman")

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f")
        plt.title("Spearman Correlation Heatmap: AIS Features")
        plt.tight_layout()

        output_path = self.output_dir / "correlation_heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved to {output_path}")
        return str(output_path)

    def plot_feature_importance(
        self, df: pd.DataFrame, target: str = "conflict_label",
        features: Optional[list] = None
    ) -> str:
        """
        Feature importance bar chart using Random Forest or XGBoost.
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("sklearn/matplotlib not installed.")
            return ""

        logger.info("Plotting feature importance...")
        if features is None:
            features = [
                "sog", "cog", "rot_abs", "is_dark_ship", "military_ratio",
                "dark_ship_ratio", "mean_rot_abs", "loitering_density",
            ]

        available = [f for f in features if f in df.columns]
        if len(available) < 2:
            logger.warning("Not enough features for importance.")
            return ""

        X = df[available].fillna(0)
        y = df[target].fillna(0)

        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)

        importances = pd.DataFrame({
            "feature": available,
            "importance": rf.feature_importances_
        }).sort_values("importance", ascending=False)

        plt.figure(figsize=(10, 6))
        plt.barh(importances["feature"][:20], importances["importance"][:20])
        plt.xlabel("Importance")
        plt.title("Top 20 Feature Importance (Random Forest)")
        plt.gca().invert_yaxis()
        plt.tight_layout()

        output_path = self.output_dir / "feature_importance.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved to {output_path}")
        return str(output_path)

    def plot_rot_distribution(self, df: pd.DataFrame) -> str:
        """
        Violin plots of ROT distribution pre/post conflict.
        """
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("seaborn/matplotlib not installed.")
            return ""

        logger.info("Plotting ROT distribution...")
        if "rot_abs" not in df.columns or "conflict_label" not in df.columns:
            logger.warning("Required columns not found. Skipping.")
            return ""

        df = df.dropna(subset=["rot_abs", "conflict_label"]).copy()
        df["conflict_status"] = df["conflict_label"].map({0: "Pre/No Conflict", 1: "During Conflict"})

        plt.figure(figsize=(10, 6))
        sns.violinplot(data=df, x="conflict_status", y="rot_abs", inner="quartile")
        plt.xlabel("Conflict Status")
        plt.ylabel("|ROT| (deg/min)")
        plt.title("ROT Distribution: Pre vs. During Conflict")
        plt.tight_layout()

        output_path = self.output_dir / "rot_distribution_violin.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved to {output_path}")
        return str(output_path)

    def plot_destination_change_rate(self, df: pd.DataFrame) -> str:
        """
        Pre/post conflict comparison of destination change rate.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib not installed.")
            return ""

        logger.info("Plotting destination change rate...")
        if "dest_changed" not in df.columns or "conflict_label" not in df.columns:
            logger.warning("Required columns not found. Skipping.")
            return ""

        df = df.dropna(subset=["dest_changed", "conflict_label"]).copy()

        rates = (
            df.groupby("conflict_label")["dest_changed"]
            .mean()
            .reset_index()
        )
        rates["label"] = rates["conflict_label"].map({0: "Pre/No Conflict", 1: "During Conflict"})

        plt.figure(figsize=(8, 5))
        plt.bar(rates["label"], rates["dest_changed"], color=["blue", "red"])
        plt.ylabel("Destination Change Rate")
        plt.title("Destination Change Rate: Pre vs. During Conflict")
        plt.tight_layout()

        output_path = self.output_dir / "destination_change_rate.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved to {output_path}")
        return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Generate statistical visualizations")
    parser.add_argument("--input", required=True, help="Input parquet file (features)")
    parser.add_argument("--output-dir", default="outputs/figures/statistical",
                        help="Output directory for figures")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    df = pd.read_parquet(args.input)
    logger.info("Loaded {len(df):,} records from {args.input}")

    viz = StatisticalVisualizer(output_dir=args.output_dir)

    # Generate key statistical plots
    viz.plot_rot_distribution(df)
    viz.plot_destination_change_rate(df)
    viz.plot_correlation_heatmap(df)
    viz.plot_feature_importance(df)

    # Plot distributions for key features
    for feature in ["sog", "rot_abs", "dark_ship_ratio"]:
        if feature in df.columns:
            viz.plot_distribution_comparison(df, feature)

    print(f"Statistical visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()

