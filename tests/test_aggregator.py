"""Tests for AISAggregator module."""
import pandas as pd
from src.preprocessing.aggregator import AISAggregator


def _sample_feature_df() -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2024-03-01 00:10:00", "2024-03-01 01:20:00",
            "2024-03-01 06:00:00", "2024-03-01 06:30:00",
        ], utc=True),
        "mmsi": [111111111, 222222222, 111111111, 333333333],
        "lat": [56.1, 56.2, 56.1, 56.7],
        "lon": [12.1, 12.4, 12.1, 12.7],
        "mobile_type": ["Class A", "Class B", "Class A", "Class A"],
        "is_dark_ship": [0, 1, 0, 0],
        "ship_type_code": [70, 35, 30, 80],
        "sog": [10.0, 12.0, 11.0, 13.0],
        "rot_abs": [0.2, 0.5, 0.1, 0.0],
        "loitering_flag": [0, 1, 0, 0],
        "not_under_command": [0, 0, 1, 0],
        "evasive_maneuver": [0, 1, 0, 0],
        "dest_is_conflict_port": [0, 1, 0, 0],
    })


def test_aggregate_creates_expected_columns():
    aggregator = AISAggregator()
    result = aggregator.aggregate(_sample_feature_df())
    assert len(result) >= 1
    assert "traffic_count" in result.columns
    assert "class_a_ratio" in result.columns
    assert "dark_ship_ratio" in result.columns
    assert "military_ratio" in result.columns
    assert "tanker_ratio" in result.columns


def test_run_writes_output(tmp_path):
    input_path = tmp_path / "clean_input.parquet"
    output_path = tmp_path / "agg_output.parquet"
    _sample_feature_df().to_parquet(input_path, index=False)

    aggregator = AISAggregator(output_path=str(output_path))
    result = aggregator.run(str(input_path))
    assert len(result) > 0
    assert output_path.exists()
