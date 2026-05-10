"""
Statistical Conflict Correlation Framework
=========================================
Methods:
  1. Granger Causality Test   ??AIS anomaly index ??conflict intensity
  2. Cross-Correlation (CCF)  ??optimal lead window (days)
  3. Difference-in-Differences ??treatment (conflict) vs. control (N. Sea / Baltic)
  4. Event Study              ??짹30-day ATV (Abnormal Traffic Volume)
  5. Interrupted Time Series  ??OLS level + slope change at conflict onset

Key hypothesis tests per conflict zone:
  H1: Traffic volume declines significantly post-conflict onset
  H2: Dark ship ratio increases before conflict onset  (Granger lead)
  H3: Military/SAR vessel ratio increases pre-conflict
  H4: Mean |ROT| increases pre-conflict (heightened maneuver intensity)
  H5: Destination diversity drops (rerouting away from conflict zones)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse
from typing import Optional

logger = logging.getLogger(__name__)

CONFLICT_ONSET = {
    "ukraine_war":      "2022-02-24",
    "houthi_crisis":    "2023-11-19",
    "pla_taiwan_drill": "2022-08-04",
    "kerch_bridge":     "2022-10-08",
}

CONTROL_ZONES = ["north_sea", "baltic_sea"]


class CorrelationAnalyzer:
    """Perform statistical correlation analysis between AIS indicators
    and conflict events."""

    def __init__(self, output_dir: str = "outputs/tables"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[dict] = []

    def granger_causality_test(
        self, df: pd.DataFrame, feature: str, conflict_col: str = "conflict_label",
        max_lag: int = 7
    ) -> dict:
        """
        Perform Granger causality test.
        H0: feature does NOT Granger-cause conflict.
        """
        try:
            from statsmodels.tsa.stattools import grangercausalitytests
        except ImportError:
            logger.error("statsmodels not installed. Skipping Granger test.")
            return {}

        logger.info("Running Granger test: {feature} ??{conflict_col}...")

        # Prepare time series data
        df_ts = df.copy()
        if "time_bucket" not in df_ts.columns:
            df_ts["time_bucket"] = df_ts["timestamp"].dt.floor("6h")

        # Aggregate: mean of feature, sum of conflict
        g = df_ts.groupby("time_bucket")
        agg = pd.DataFrame({
            "feature_val": g[feature].mean(),
            "conflict_val": g[conflict_col].sum(),
        }).dropna()

        if len(agg) < max_lag + 10:
            logger.warning("Not enough data for Granger test.")
            return {}

        data = agg[["feature_val", "conflict_val"]].values

        try:
            result = grangercausalitytests(data, maxlag=max_lag)
            p_values = {lag: result[lag][0]["ssr_ftest"][1] for lag in result.keys()}
            min_p = min(p_values.values())
            best_lag = [k for k, v in p_values.items() if v == min_p][0]

            result_dict = {
                "test": "granger",
                "feature": feature,
                "best_lag": best_lag,
                "p_value": min_p,
                "significant": min_p < 0.05,
            }
            self.results.append(result_dict)
            logger.info("Granger test result: p={min_p:.4f} at lag {best_lag}")
            return result_dict
        except Exception as e:
            logger.error("Granger test failed: {e}")
            return {}

    def cross_correlation(
        self, df: pd.DataFrame, feature: str, conflict_col: str = "conflict_label",
        max_lags: int = 30
    ) -> dict:
        """
        Compute cross-correlation function (CCF) to find optimal lead/lag.
        Positive lag: feature leads conflict.
        """
        logger.info("Computing CCF: {feature} vs {conflict_col}...")

        df_ts = df.copy()
        if "time_bucket" not in df_ts.columns:
            df_ts["time_bucket"] = df_ts["timestamp"].dt.floor("6H")

        agg = (
            df_ts.groupby("time_bucket")
            .agg(
                feature_val=(feature, "mean"),
                conflict_val=(conflict_col, "sum"),
            )
            .dropna()
        )

        if len(agg) < max_lags * 2:
            logger.warning("Not enough data for CCF.")
            return {}

        feature_series = agg["feature_val"].values - agg["feature_val"].mean()
        conflict_series = agg["conflict_val"].values - agg["conflict_val"].mean()

        correlations = {}
        for lag in range(-max_lags, max_lags + 1):
            if lag < 0:
                corr = np.corrcoef(feature_series[:lag], conflict_series[-lag:])[0, 1]
            elif lag > 0:
                corr = np.corrcoef(feature_series[lag:], conflict_series[:-lag])[0, 1]
            else:
                corr = np.corrcoef(feature_series, conflict_series)[0, 1]
            correlations[lag] = corr if not np.isnan(corr) else 0.0

        best_lag = max(correlations, key=lambda k: correlations.get(k, 0))
        result = {
            "test": "ccf",
            "feature": feature,
            "best_lag_days": best_lag,
            "max_corr": correlations[best_lag],
            "correlations": correlations,
        }
        self.results.append(result)
        logger.info(
            f"CCF: best lag = {best_lag} days, "
            f"corr = {correlations[best_lag]:.4f}"
        )
        return result

    def difference_in_differences(
        self, df: pd.DataFrame, feature: str, onset_date: str,
        treatment_zone: str, control_zones: Optional[list] = None
    ) -> dict:
        """
        Difference-in-Differences analysis.
        Compares feature change in treatment vs control zones.
        """
        logger.info("Running DiD analysis for {feature}...")

        if control_zones is None:
            control_zones = CONTROL_ZONES

        df = df.copy()
        onset = pd.Timestamp(onset_date)

        # Define treatment and control groups
        treat = df[df["conflict_zone_name"] == treatment_zone].copy()
        control = df[df["conflict_zone_name"].isin(control_zones)].copy()

        if len(treat) == 0 or len(control) == 0:
            logger.warning("Not enough data for DiD.")
            return {}

        treat["group"] = "treatment"
        control["group"] = "control"

        combined = pd.concat([treat, control], ignore_index=True)
        combined["post"] = (combined["timestamp"] >= onset).astype(int)

        # DiD regression: feature ~ group + post + group:post
        try:
            import statsmodels.api as sm

            y = combined[feature].fillna(combined[feature].median())
            X = sm.add_constant(pd.get_dummies(combined[["group", "post"]]))
            X["group_treatment:post"] = X["group_treatment"] * combined["post"]

            model = sm.OLS(y, X).fit()

            result = {
                "test": "did",
                "feature": feature,
                "treatment_zone": treatment_zone,
                "did_coef": model.params.get("group_treatment:post", np.nan),
                "did_p_value": model.pvalues.get("group_treatment:post", np.nan),
                "significant": model.pvalues.get("group_treatment:post", 1) < 0.05,
            }
            self.results.append(result)
            logger.info(
                f"DiD result: coef={result['did_coef']:.4f}, "
                f"p={result['did_p_value']:.4f}"
            )
            return result
        except ImportError:
            logger.error("statsmodels not installed.")
            return {}
        except Exception as e:
            logger.error("DiD failed: {e}")
            return {}

    def event_study(
        self, df: pd.DataFrame, feature: str, onset_date: str,
        window: int = 30
    ) -> dict:
        """
        Event study: analyze feature behavior 짹window days around conflict onset.
        Computes Abnormal Traffic Volume (ATV).
        """
        logger.info("Running event study for {feature}...")

        onset = pd.Timestamp(onset_date)
        df = df.copy()
        df["days_to_event"] = (df["timestamp"] - onset).dt.days

        # Filter to event window
        event_data = df[df["days_to_event"].between(-window, window)].copy()

        if len(event_data) == 0:
            logger.warning("No data in event window.")
            return {}

        # Compute average feature value by day relative to event
        daily_avg = event_data.groupby("days_to_event")[feature].mean().reset_index()

        # Baseline: days -window to -10
        baseline = daily_avg[daily_avg["days_to_event"] < -10][feature].mean()
        daily_avg["atv"] = daily_avg[feature] - baseline

        result = {
            "test": "event_study",
            "feature": feature,
            "onset_date": onset_date,
            "window": window,
            "baseline": baseline,
            "max_atv": daily_avg["atv"].max(),
            "min_atv": daily_avg["atv"].min(),
            "daily_avg": daily_avg,
        }
        self.results.append(result)
        logger.info(
            f"Event study: baseline={baseline:.4f}, "
            f"max ATV={result['max_atv']:.4f}"
        )
        return result

    def interrupted_time_series(
        self, df: pd.DataFrame, feature: str, onset_date: str
    ) -> dict:
        """
        Interrupted Time Series (ITS) analysis.
        Tests for level and slope changes at intervention (conflict onset).
        """
        logger.info("Running ITS analysis for {feature}...")

        onset = pd.Timestamp(onset_date)
        df = df.copy()

        if "time_bucket" not in df.columns:
            df["time_bucket"] = df["timestamp"].dt.floor("6H")

        agg = df.groupby("time_bucket")[feature].mean().reset_index()
        agg = agg.dropna()

        if len(agg) < 20:
            logger.warning("Not enough data for ITS.")
            return {}

        agg["days_from_onset"] = (agg["time_bucket"] - onset).dt.days
        agg["post_onset"] = (agg["days_from_onset"] > 0).astype(int)
        agg["trend_post"] = agg["days_from_onset"] * agg["post_onset"]

        try:
            import statsmodels.api as sm

            X = sm.add_constant(agg[["days_from_onset", "post_onset", "trend_post"]])
            model = sm.OLS(agg[feature].fillna(agg[feature].median()), X).fit()

            result = {
                "test": "its",
                "feature": feature,
                "onset_date": onset_date,
                "level_change": model.params.get("post_onset", np.nan),
                "level_p_value": model.pvalues.get("post_onset", np.nan),
                "slope_change": model.params.get("trend_post", np.nan),
                "slope_p_value": model.pvalues.get("trend_post", np.nan),
                "significant_level": model.pvalues.get("post_onset", 1) < 0.05,
                "significant_slope": model.pvalues.get("trend_post", 1) < 0.05,
            }
            self.results.append(result)
            logger.info(
                f"ITS: level_change={result['level_change']:.4f}, "
                f"slope_change={result['slope_change']:.4f}"
            )
            return result
        except ImportError:
            logger.error("statsmodels not installed.")
            return {}
        except Exception as e:
            logger.error("ITS failed: {e}")
            return {}

    def save_results(self) -> str:
        """Save all test results to CSV."""
        if not self.results:
            logger.warning("No results to save.")
            return ""

        output_path = self.output_dir / "correlation_analysis_results.csv"

        # Flatten results for CSV
        flat_results = []
        for r in self.results:
            flat = {
                k: v
                for k, v in r.items()
                if not isinstance(v, (dict, pd.DataFrame))
            }
            flat_results.append(flat)

        results_df = pd.DataFrame(flat_results)
        results_df.to_csv(output_path, index=False)
        logger.info("Saved results to {output_path}")
        return str(output_path)

    def run_all_tests(
        self, df: pd.DataFrame, features: Optional[list] = None,
        onset_date: str = "2022-02-24", treatment_zone: str = "black_sea"
    ) -> list:
        """
        Run all correlation tests for specified features.
        """
        if features is None:
            features = ["traffic_count", "dark_ship_ratio", "military_ratio",
                       "mean_rot_abs", "loitering_density"]

        for feature in features:
            if feature not in df.columns:
                logger.warning("Feature {feature} not in DataFrame. Skipping.")
                continue

            # Granger test
            self.granger_causality_test(df, feature)

            # Cross-correlation
            self.cross_correlation(df, feature)

            # Event study
            self.event_study(df, feature, onset_date)

            # ITS
            self.interrupted_time_series(df, feature, onset_date)

        # DiD for key features
        for feature in ["traffic_count", "dark_ship_ratio"]:
            if feature in df.columns:
                self.difference_in_differences(df, feature, onset_date, treatment_zone)

        return self.results


def main():
    parser = argparse.ArgumentParser(
        description="Run correlation analysis"
    )
    parser.add_argument(
        "--input", required=True,
        help="Input parquet file (features or aggregated)"
    )
    parser.add_argument(
        "--output", default="outputs/tables",
        help="Output directory"
    )
    parser.add_argument(
        "--onset", default="2022-02-24",
        help="Conflict onset date"
    )
    parser.add_argument(
        "--zone", default="black_sea",
        help="Treatment zone"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    df = pd.read_parquet(args.input)
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    if "time_bucket" in df.columns and df["time_bucket"].dt.tz is not None:
        df["time_bucket"] = df["time_bucket"].dt.tz_localize(None)
    logger.info("Loaded {len(df):,} records from {args.input}")

    analyzer = CorrelationAnalyzer(output_dir=args.output)
    analyzer.run_all_tests(df, onset_date=args.onset, treatment_zone=args.zone)
    analyzer.save_results()

    print(f"Analysis complete. Results saved to {args.output}")


if __name__ == "__main__":
    main()

