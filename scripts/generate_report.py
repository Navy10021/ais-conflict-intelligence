"""
Generate Final Research Report
===========================
Compile all results into a publication-ready PDF report.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse
from datetime import datetime

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate final research report from all analysis results."""

    PAPER_STRUCTURE = {
        "title": "Maritime Traffic Anomaly Detection as a Precursor to Armed Conflict: Evidence from AIS Data in Global Hotspots (2022-2024)",
        "sections": [
            "1. Introduction",
            "  1.1 Motivation ??AIS as a conflict early-warning signal",
            "  1.2 Research questions and hypotheses (H1-H5)",
            "  1.3 Dataset & contributions (aisdk + ACLED)",
            "2. Background and Related Work",
            "  2.1 AIS system architecture and data characteristics",
            "  2.2 AIS limitations: spooping, dark shipping, Class B gaps",
            "  2.3 Prior work: maritime anomaly detection",
            "  2.4 Prior work: conflict early-warning systems",
            "3. Data and Methodology",
            "  3.1 aisdk multi-day data pipeline",
            "  3.2 Conflict event data (ACLED integration)",
            "  3.3 Feature engineering ??six categories including ROT & Destination",
            "  3.4 Analytical framework (Granger, DiD, ITS, Event Study)",
            "4. Empirical Results",
            "  4.1 Black Sea (Russia-Ukraine War) ??traffic collapse, dark ship surge",
            "  4.2 Red Sea (Houthi Crisis) ??tanker rerouting, destination shift",
            "  4.3 Taiwan Strait ??PLA exercise maneuver intensity (ROT)",
            "  4.4 Leading indicator analysis ??Granger test, optimal lead-lag",
            "5. Predictive Modeling",
            "  5.1 Model comparison ??LSTM vs. TFT vs. XGBoost",
            "  5.2 Zone-level performance (AUROC, F2, Lead Time)",
            "  5.3 SHAP analysis ??ROT & dark_ship_ratio as dominant predictors",
            "  5.4 Horizon sensitivity ??T+3 to T+30",
            "6. Discussion",
            "  6.1 AIS data limitations and validity threats",
            "  6.2 Operational implications for maritime conflict early warning",
            "7. Conclusion",
            "References",
            "Appendix A ??Supplementary Figures",
            "Appendix B ??Full Statistical Tables",
            "Appendix C ??aisdk Schema Documentation",
        ],
    }

    def __init__(self, output_dir: str = "outputs/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.report_content = []

    def collect_results(self) -> dict:
        """Collect all results from outputs."""
        results = {
            "tables": {},
            "figures": {},
            "models": {},
        }

        # Load tables
        tables_dir = Path("outputs/tables")
        if tables_dir.exists():
            for csv_file in tables_dir.glob("*.csv"):
                try:
                    df = pd.read_csv(csv_file)
                    results["tables"][csv_file.stem] = df
                except Exception as e:
                    logger.warning("Failed to load {csv_file}: {e}")

        # List figures
        figures_dir = Path("outputs/figures")
        if figures_dir.exists():
            for fig_file in figures_dir.rglob("*.png"):
                results["figures"][fig_file.stem] = str(fig_file)

        # Load model results
        models_dir = Path("outputs/models")
        if models_dir.exists():
            for csv_file in models_dir.rglob("*.csv"):
                try:
                    df = pd.read_csv(csv_file)
                    results["models"][csv_file.stem] = df
                except Exception as e:
                    logger.warning("Failed to load {csv_file}: {e}")

        return results

    def generate_summary_statistics(self, df: pd.DataFrame) -> str:
        """Generate summary statistics section."""
        lines = []

        lines.append("## Dataset Summary\n")
        lines.append(f"- Total records: {len(df):,}")
        lines.append(f"- Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
        lines.append(f"- Unique vessels (MMSI): {df['mmsi'].nunique():,}")

        if "ship_type_code" in df.columns:
            type_counts = df.groupby("ship_type_code")["mmsi"].nunique().to_dict()
            lines.append(f"- Military vessels: {type_counts.get(35, 0):,}")
            lines.append(f"- Fishing vessels: {type_counts.get(30, 0):,}")
            lines.append(f"- Cargo vessels: {type_counts.get(70, 0):,}")

        if "in_conflict_zone" in df.columns:
            conflict_vessels = df[df["in_conflict_zone"] == True]["mmsi"].nunique()
            lines.append(f"- Vessels in conflict zones: {conflict_vessels:,}")

        return "\n".join(lines)

    def generate_hypothesis_results(self, results: dict) -> str:
        """Generate hypothesis testing results."""
        lines = []

        lines.append("## Hypothesis Testing Results\n")

        # H1-H5 from correlation analysis
        if "correlation_analysis_results" in results.get("tables", {}):
            df = results["tables"]["correlation_analysis_results"]
            lines.append("### Statistical Tests Summary\n")
            for _, row in df.iterrows():
                lines.append(f"- **{row.get('test', 'N/A')}** ({row.get('feature', 'N/A')}):")
                lines.append(f"  - p-value: {row.get('p_value', 'N/A'):.4f}")
                lines.append(f"  - Significant: {row.get('significant', False)}")

        return "\n".join(lines)

    def generate_model_performance(self, results: dict) -> str:
        """Generate model performance section."""
        lines = []

        lines.append("## Model Performance\n")

        if "prediction_results" in results.get("tables", {}):
            df = results["tables"]["prediction_results"]
            lines.append("### Prediction Results\n")
            lines.append(df.to_markdown(index=False))

        if "evaluation_results" in results.get("tables", {}):
            df = results["tables"]["evaluation_results"]
            lines.append("\n### Evaluation Metrics\n")
            for _, row in df.iterrows():
                lines.append(f"- **{row.get('model', 'N/A')}** (T+{row.get('horizon', 'N/A')}):")
                lines.append(f"  - AUROC: {row.get('auroc', 0):.4f}")
                lines.append(f"  - AUPRC: {row.get('auprc', 0):.4f}")
                lines.append(f"  - F2-Score: {row.get('f2_score', 0):.4f}")

        return "\n".join(lines)

    def generate_markdown_report(self, df: pd.DataFrame, results: dict) -> str:
        """Generate full markdown report."""
        lines = []

        # Title
        lines.append(f"# {self.PAPER_STRUCTURE['title']}\n")
        lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Abstract (placeholder)
        lines.append("## Abstract\n")
        lines.append("This study investigates maritime behavioral anomalies...")
        lines.append("")

        # Summary statistics
        lines.append(self.generate_summary_statistics(df))
        lines.append("")

        # Hypothesis results
        lines.append(self.generate_hypothesis_results(results))
        lines.append("")

        # Model performance
        lines.append(self.generate_model_performance(results))
        lines.append("")

        # Paper structure (outline)
        lines.append("## Paper Structure Outline\n")
        for section in self.PAPER_STRUCTURE["sections"]:
            lines.append(f"{section}")

        return "\n".join(lines)

    def save_report(self, markdown_content: str) -> str:
        """Save report as Markdown and attempt PDF conversion."""
        md_path = self.output_dir / "mcis_final_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        logger.info("Saved Markdown report to {md_path}")

        # Try to convert to PDF
        pdf_path = self.output_dir / "mcis_final_report.pdf"
        try:
            import subprocess
            result = subprocess.run(
                ["pandoc", str(md_path), "-o", str(pdf_path)],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                logger.info("Saved PDF report to {pdf_path}")
                return str(pdf_path)
            else:
                logger.warning("Pandoc conversion failed: {result.stderr}")
        except FileNotFoundError:
            logger.warning("Pandoc not installed. PDF not generated.")

        return str(md_path)


def main():
    parser = argparse.ArgumentParser(description="Generate final report")
    parser.add_argument("--input", required=True, help="Input parquet file (features)")
    parser.add_argument("--output-dir", default="outputs/reports")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    df = pd.read_parquet(args.input)
    logger.info("Loaded {len(df):,} records")

    generator = ReportGenerator(output_dir=args.output_dir)
    results = generator.collect_results()

    markdown = generator.generate_markdown_report(df, results)
    output_path = generator.save_report(markdown)

    print(f"Report generated: {output_path}")


if __name__ == "__main__":
    main()

