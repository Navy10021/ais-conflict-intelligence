"""Tests for AISCleaner module."""
import pytest
import pandas as pd
import numpy as np
from src.preprocessing.cleaner import AISCleaner, NAV_STATUS_MAP, SHIP_TYPE_MAP, SPECIAL_MMSI


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2024-03-01 00:00:00",
            "2024-03-01 01:00:00",
            "2024-03-01 02:00:00",
            "2024-03-01 03:00:00",
        ]),
        "mobile_type": ["Class A", "Class B", "Base Station", "Class A"],
        "mmsi": [123456789, 987654321, 111111111, 222222222],
        "lat": [56.0, 57.0, 91.0, 56.5],
        "lon": [12.0, 13.0, 181.0, 12.5],
        "nav_status": ["Under way using engine", "At anchor", "Moored", "Not under command"],
        "rot": [0.0, 5.0, np.nan, 10.0],
        "sog": [10.0, 5.0, 102.3, 8.0],
        "cog": [180.0, 90.0, 360.0, 270.0],
        "heading": [180, 90, 511, 270],
        "ship_type": ["Cargo", "Fishing", "Pleasure", "Tanker"],
        "vessel_name": ["VESSEL_1", "VESSEL_2", "Unknown", "VESSEL_3"],
        "imo": ["1234567", "Unknown", "Unknown", "7654321"],
        "callsign": ["ABC123", "Unknown", "Unknown", "XYZ789"],
        "destination": ["COPENHAGEN", "Unknown", "Unknown", "STOCKHOLM"],
        "eta": ["02/03/2024 12:00:00", "Unknown", "Unknown", "03/03/2024 10:00:00"],
        "ant_bow": [10.0, 5.0, 3.0, 8.0],
        "ant_stern": [190.0, 45.0, 20.0, 150.0],
        "ant_port": [5.0, 3.0, 2.0, 4.0],
        "ant_starboard": [5.0, 3.0, 2.0, 4.0],
        "length": [np.nan, np.nan, 25.0, np.nan],
        "width": [np.nan, np.nan, 8.0, np.nan],
        "draught": [10.0, 5.0, 3.0, 8.0],
    })


class TestAISCleaner:
    """Test cases for AISCleaner."""

    def test_init(self):
        """Test cleaner initialization."""
        from pathlib import Path
        cleaner = AISCleaner(output_path="test_output.parquet")
        assert cleaner.output_path == Path("test_output.parquet")
        assert cleaner.report == {}

    def test_filter_mobile_type(self, sample_df):
        """Test mobile type filtering."""
        cleaner = AISCleaner()
        result = cleaner.filter_mobile_type(sample_df)
        
        # Should remove Base Station
        assert len(result) == 3
        assert "Base Station" not in result["mobile_type"].values

    def test_filter_mmsi(self, sample_df):
        """Test MMSI filtering."""
        cleaner = AISCleaner()
        result = cleaner.filter_mmsi(sample_df)
        
        # Should keep only 200M-799M range
        assert all(result["mmsi"].between(200_000_000, 799_999_999))

    def test_filter_coordinates(self, sample_df):
        """Test coordinate filtering."""
        cleaner = AISCleaner()
        result = cleaner.filter_coordinates(sample_df)
        
        # Should remove sentinel values (91.0, 181.0)
        assert len(result) == 3
        assert 91.0 not in result["lat"].values
        assert 181.0 not in result["lon"].values

    def test_clean_kinematics(self, sample_df):
        """Test kinematic sentinel cleaning."""
        cleaner = AISCleaner()
        result = cleaner.clean_kinematics(sample_df.copy())
        
        # SOG=102.3 should become NaN
        assert pd.isna(result.loc[2, "sog"]) or result.loc[2, "sog"] != 102.3
        
        # COG=360 should become NaN
        assert pd.isna(result.loc[2, "cog"]) or result.loc[2, "cog"] != 360.0

    def test_filter_timestamps(self, sample_df):
        """Test timestamp filtering."""
        cleaner = AISCleaner()
        result = cleaner.filter_timestamps(sample_df)
        
        # Should be sorted by mmsi and timestamp
        # Check that within each MMSI group, timestamps are increasing
        for mmsi, group in result.groupby("mmsi"):
            assert group["timestamp"].is_monotonic_increasing, f"Timestamps not monotonic for MMSI {mmsi}"

    def test_encode_nav_status(self, sample_df):
        """Test navigational status encoding."""
        cleaner = AISCleaner()
        result = cleaner.encode_nav_status(sample_df.copy())
        
        assert "nav_status_code" in result.columns
        assert result["nav_status_code"].iloc[0] == 0  # Under way using engine
        assert result["nav_status_code"].iloc[1] == 1  # At anchor

    def test_encode_ship_type(self, sample_df):
        """Test ship type encoding."""
        cleaner = AISCleaner()
        result = cleaner.encode_ship_type(sample_df.copy())
        
        assert "ship_type_code" in result.columns
        assert result["ship_type_code"].iloc[0] == 70  # Cargo
        assert result["ship_type_code"].iloc[1] == 30  # Fishing

    def test_impute_dimensions_from_antenna(self, sample_df):
        """Test dimension imputation from antenna offsets."""
        cleaner = AISCleaner()
        result = cleaner.impute_dimensions_from_antenna(sample_df.copy())
        
        # Length = A + B, Width = C + D
        assert result["length"].iloc[0] == 200.0  # 10 + 190
        assert result["width"].iloc[0] == 10.0    # 5 + 5

    def test_deduplicate(self, sample_df):
        """Test deduplication."""
        # Add duplicate row
        df_with_dup = pd.concat([sample_df, sample_df.iloc[[0]]], ignore_index=True)
        
        cleaner = AISCleaner()
        result = cleaner.deduplicate(df_with_dup)
        
        assert len(result) == len(sample_df)

    def test_run_pipeline(self, sample_df, tmp_path):
        """Test full cleaning pipeline."""
        output_path = tmp_path / "cleaned.parquet"
        cleaner = AISCleaner(output_path=str(output_path))
        
        result = cleaner.run(sample_df)
        
        assert len(result) > 0
        assert output_path.exists()


class TestNAV_STATUS_MAP:
    """Test navigational status mapping."""

    def test_all_status_codes(self):
        """Test that all expected statuses are mapped."""
        expected = {
            "Under way using engine": 0,
            "At anchor": 1,
            "Not under command": 2,
            "Restricted maneuverability": 3,
            "Constrained by her draught": 4,
            "Moored": 5,
            "Aground": 6,
            "Engaged in fishing": 7,
            "Under way sailing": 8,
        }
        for key, value in expected.items():
            assert NAV_STATUS_MAP[key] == value


class TestSHIP_TYPE_MAP:
    """Test ship type mapping."""

    def test_major_ship_types(self):
        """Test that major ship types are mapped."""
        assert SHIP_TYPE_MAP["Cargo"] == 70
        assert SHIP_TYPE_MAP["Tanker"] == 80
        assert SHIP_TYPE_MAP["Fishing"] == 30
        assert SHIP_TYPE_MAP["Military"] == 35


class TestSPECIAL_MMSI:
    """Test special MMSI ranges."""

    def test_coastal_bs_range(self):
        """Test coastal base station range."""
        lo, hi = SPECIAL_MMSI["coastal_bs"]
        assert lo == 0
        assert hi == 99_999_999

    def test_sar_aircraft_range(self):
        """Test SAR aircraft range."""
        lo, hi = SPECIAL_MMSI["sar_aircraft"]
        assert lo == 111_000_000
        assert hi == 111_999_999
