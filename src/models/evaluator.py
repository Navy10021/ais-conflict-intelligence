"""
Model Evaluation & Reporting
==========================
Evaluation metrics and reporting for conflict prediction models.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluate model performance with multiple metrics.
    """

    HORIZONS = [3, 7, 14, 30]  # T+days

    def __init__(self, output_dir: str = "outputs/models/evaluation"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[dict] = []

    def compute_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray
    ) -> dict:
        """Compute evaluation metrics."""
        try:
            from sklearn.metrics import (
                roc_auc_score, average_precision_score,
                fbeta_score, precision_score, recall_score,
                confusion_matrix
            )
        except ImportError:
            logger.error("scikit-learn not installed.")
            return {}

        metrics = {}

        # AUROC
        try:
            metrics["auroc"] = roc_auc_score(y_true, y_prob)
        except Exception:
            metrics["auroc"] = np.nan

        # AUPRC
        try:
            metrics["auprc"] = average_precision_score(y_true, y_prob)
        except Exception:
            metrics["auprc"] = np.nan

        # F2-Score (emphasizes recall)
        try:
            metrics["f2_score"] = fbeta_score(y_true, y_pred, beta=2)
        except Exception:
            metrics["f2_score"] = np.nan

        # Precision & Recall
        try:
            metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
            metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
        except Exception:
            metrics["precision"] = np.nan
            metrics["recall"] = np.nan

        # Confusion Matrix
        try:
            cm = confusion_matrix(y_true, y_pred)
            metrics["tn"] = int(cm[0, 0]) if cm.shape[0] > 1 else 0
            metrics["fp"] = int(cm[0, 1]) if cm.shape[0] > 1 else 0
            metrics["fn"] = int(cm[1, 0]) if cm.shape[0] > 1 else 0
            metrics["tp"] = int(cm[1, 1]) if cm.shape[0] > 1 else 0
        except Exception:
            pass

        return metrics

    def evaluate_horizon(
        self, y_true: pd.Series, y_pred: pd.DataFrame,
        horizon: int, model_name: str = "model"
    ) -> dict:
        """Evaluate model at specific horizon."""
        logger.info("Evaluating {model_name} at T+{horizon}...")

        if horizon not in y_pred.columns:
            logger.warning("Horizon T+{horizon} not in predictions.")
            return {}

        y_pred_h = y_pred[horizon].values
        y_true_h = y_true.values

        # Convert probabilities to binary predictions
        y_pred_binary = (y_pred_h > 0.5).astype(int)

        metrics = self.compute_metrics(y_true_h, y_pred_binary, y_pred_h)
        metrics["model"] = model_name
        metrics["horizon"] = horizon

        self.results.append(metrics)
        logger.info(
            f"T+{horizon} - AUROC: {metrics.get('auroc', 0):.4f}, "
            f"F2: {metrics.get('f2_score', 0):.4f}"
        )
        return metrics

    def compute_lead_time(
        self, y_true: pd.Series, y_prob: pd.DataFrame,
        threshold: float = 0.5
    ) -> float:
        """Compute mean lead time (days before conflict)."""
        logger.info("Computing mean lead time...")

        lead_times = []
        for _, row in y_prob.iterrows():
            pred_days = None
            for days in sorted(row.index):
                if row[days] > threshold:
                    pred_days = days
                    break
            if pred_days is not None and y_true.iloc[_] == 1:
                lead_times.append(pred_days)

        mean_lead = np.mean(lead_times) if lead_times else np.nan
        logger.info("Mean lead time: {mean_lead:.2f} days")
        return mean_lead

    def compute_false_alarm_rate(
        self, y_true: pd.Series, y_pred: pd.DataFrame,
        threshold: float = 0.5
    ) -> float:
        """Compute false alarm rate."""
        logger.info("Computing false alarm rate...")

        y_pred_binary = (y_pred > threshold).astype(int)
        total_predictions = len(y_true)
        false_alarms = (
            (y_pred_binary.sum(axis=1) > 0) & (y_true == 0)
        ).sum()

        far = false_alarms / total_predictions if total_predictions > 0 else np.nan
        logger.info("False alarm rate: {far:.4f}")
        return far

    def compare_models(
        self, results_dict: dict
    ) -> pd.DataFrame:
        """Compare multiple models."""
        logger.info("Comparing models...")

        comparison = []
        for model_name, horizons in results_dict.items():
            for horizon, metrics in horizons.items():
                row = {"model": model_name, "horizon": horizon}
                row.update(metrics)
                comparison.append(row)

        df = pd.DataFrame(comparison)
        output_path = self.output_dir / "model_comparison.csv"
        df.to_csv(output_path, index=False)
        logger.info("Saved comparison to {output_path}")

        return df

    def save_results(self) -> str:
        """Save all evaluation results."""
        if not self.results:
            logger.warning("No results to save.")
            return ""

        df = pd.DataFrame(self.results)
        output_path = self.output_dir / "evaluation_results.csv"
        df.to_csv(output_path, index=False)
        logger.info("Saved evaluation results to {output_path}")

        return str(output_path)

    def generate_report(self, df: pd.DataFrame) -> str:
        """Generate evaluation report."""
        logger.info("Generating evaluation report...")

        try:
            from jinja2 import Template
        except ImportError:
            logger.error("jinja2 not installed for report generation.")
            return ""

        template_str = """
# Model Evaluation Report

## Summary
- Total models evaluated: {{ models|length }}
- Horizons: {{ horizons|join(', ') }}

## Results by Horizon

{% for horizon in horizons %}
### T+{{ horizon }} Days
{% for r in results if r.horizon == horizon %}
- **{{ r.model }}**: AUROC={{ "%.4f"|format(r.auroc or 0) }}, F2={{ "%.4f"|format(r.f2_score or 0) }}
{% endfor %}
{% endfor %}

## Best Models
{% for horizon in horizons %}
- T+{{ horizon }}: {{ best_models[horizon] }}
{% endfor %}
"""

        results = self.results
        horizons = sorted(set(r["horizon"] for r in results))
        models = sorted(set(r["model"] for r in results))

        # Find best model per horizon by AUROC
        best_models = {}
        for h in horizons:
            h_results = [r for r in results if r["horizon"] == h]
            if h_results:
                best = max(h_results, key=lambda x: x.get("auroc", 0))
                best_models[h] = f"{best['model']} (AUROC={best.get('auroc', 0):.4f})"

        template = Template(template_str)
        report = template.render(
            results=results, horizons=horizons,
            models=models, best_models=best_models
        )

        output_path = self.output_dir / "evaluation_report.md"
        with open(output_path, "w") as f:
            f.write(report)

        logger.info("Saved report to {output_path}")
        return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate prediction models")
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument("--predictions", required=True, help="Predictions CSV")
    parser.add_argument("--output", default="outputs/models/evaluation")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    df = pd.read_parquet(args.input)
    preds = pd.read_csv(args.predictions)

    logger.info("Loaded {len(df):,} records")

    evaluator = ModelEvaluator(output_dir=args.output)
    # Example: evaluate at different horizons
    for h in [7, 14, 30]:
        if str(h) in preds.columns:
            evaluator.evaluate_horizon(
                df["conflict_label"], preds[[str(h)]],
                horizon=h, model_name="model"
            )

    evaluator.save_results()
    evaluator.generate_report(df)

    print(f"Evaluation complete. Results: {args.output}")


if __name__ == "__main__":
    main()

