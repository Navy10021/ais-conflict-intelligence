"""
Baseline Models
==============
ARIMA and Prophet baselines for conflict prediction.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse
from typing import Any

logger = logging.getLogger(__name__)


class ARIMABaseline:
    """ARIMA baseline for time series forecasting."""

    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.model = None
        self.results = None

    def fit(self, series: pd.Series):
        """Fit ARIMA model."""
        try:
            from statsmodels.tsa.arima.model import ARIMA
        except ImportError:
            logger.error("statsmodels not installed.")
            return None

        logger.info("Fiting ARIMA{self.order}...")
        try:
            self.model = ARIMA(series, order=self.order)
            self.results = self.model.fit()
            logger.info("ARIMA fit complete.")
            return self.results
        except Exception as e:
            logger.error("ARIMA fit failed: {e}")
            return None

    def predict(self, steps: int = 7) -> np.ndarray:
        """Generate forecasts."""
        if self.results is None:
            logger.error("Model not fitted.")
            return np.array([])

        try:
            forecast = self.results.forecast(steps=steps)
            return forecast
        except Exception as e:
            logger.error("Prediction failed: {e}")
            return np.array([])


class ProphetBaseline:
    """Prophet baseline for time series forecasting."""

    def __init__(self):
        self.model = None
        self.forecast_df = None

    def fit(self, df: pd.DataFrame, date_col: str = "time_bucket", target_col: str = "traffic_count"):
        """Fit Prophet model."""
        try:
            from prophet import Prophet
        except ImportError:
            logger.error("prophet not installed.")
            return None

        logger.info("Fiting Prophet model...")

        # Prepare data for Prophet
        prophet_df = df[[date_col, target_col]].copy()
        prophet_df.columns = ["ds", "y"]
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

        try:
            self.model = Prophet(
                seasonality_mode="multiplicative",
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
            )
            self.model.fit(prophet_df)
            logger.info("Prophet fit complete.")
            return self.model
        except Exception as e:
            logger.error("Prophet fit failed: {e}")
            return None

    def predict(self, periods: int = 7, freq: str = "6H") -> pd.DataFrame:
        """Generate forecasts."""
        if self.model is None:
            logger.error("Model not fitted.")
            return pd.DataFrame()

        try:
            future = self.model.make_future_dataframe(periods=periods, freq=freq)
            self.forecast_df = self.model.predict(future)
            return self.forecast_df
        except Exception as e:
            logger.error("Prediction failed: {e}")
            return pd.DataFrame()


def run_baseline_analysis(
    df: pd.DataFrame,
    target: str = "traffic_count",
    output_dir: str = "outputs/models/baseline"
) -> dict:
    """Run both baseline models and compare."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, Any]] = {}

    # Prepare time series
    if "time_bucket" not in df.columns:
        df = df.copy()
        df["time_bucket"] = df["timestamp"].dt.floor("6H")

    ts = df.groupby("time_bucket")[target].mean().dropna()

    if len(ts) < 20:
        logger.warning("Not enough data for baseline models.")
        return results

    # ARIMA
    logger.info("Running ARIMA baseline...")
    arima = ARIMABaseline()
    arima.fit(ts)
    if arima.results:
        forecast_7 = arima.predict(steps=28)  # 7 days * 4 (6H buckets)
        results["arima"] = {
            "model": "ARIMA",
            "forecast_7days": forecast_7.tolist() if len(forecast_7) > 0 else [],
        }

    # Prophet
    logger.info("Running Prophet baseline...")
    df_ts = ts.reset_index()
    prophet = ProphetBaseline()
    prophet.fit(df_ts, date_col="time_bucket", target_col=target)
    if prophet.model:
        forecast = prophet.predict(periods=28)
        results["prophet"] = {
            "model": "Prophet",
            "forecast_columns": list(forecast.columns),
        }

    # Save results
    results_df = pd.DataFrame(
        [{"model": k, "details": str(v)} for k, v in results.items()]
    )
    results_df.to_csv(output_path / "baseline_results.csv", index=False)
    logger.info("Saved baseline results to {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run baseline models")
    parser.add_argument("--input", required=True, help="Input parquet file (aggregated)")
    parser.add_argument("--target", default="traffic_count", help="Target column")
    parser.add_argument("--output", default="outputs/models/baseline")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    df = pd.read_parquet(args.input)
    logger.info("Loaded {len(df):,} records")

    results = run_baseline_analysis(df, target=args.target, output_dir=args.output)

    print(f"Baseline models complete. Results: {list(results.keys())}")


if __name__ == "__main__":
    main()

