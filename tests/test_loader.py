"""Tests for AISDKLoader module."""
import pytest
import pandas as pd
from pathlib import Path
from src.preprocessing.loader import AISDKLoader, COLUMN_RENAME, DTYPE_MAP


@pytest.fixture
def sample_csv_path(tmp_path):
    """Create a sample AIS CSV file for testing."""
    csv_dir = tmp_path / "2024-03-01"
    csv_dir.mkdir()
    csv_file = csv_dir / "aisdk-2024-03-01.csv"
    
    # Create minimal CSV with required columns
    df = pd.DataFrame({
        "# Timestamp": ["01/03/2024 00:00:00", "01/03/2024 01:00:00"],
        "Type of mobile": ["Class A", "Class B"],
        "MMSI": [123456789, 987654321],
        "Latitude": [56.0, 57.0],
        "Longitude": [12.0, 13.0],
        "Navigational status": ["Under way using engine", "At anchor"],
        "ROT": [0.0, 5.0],
        "SOG": [10.0, 5.0],
        "COG": [180.0, 90.0],
        "Heading": [180, 90],
        "IMO": ["1234567", "Unknown"],
        "Callsign": ["ABC123", "Unknown"],
        "Name": ["VESSEL_1", "Unknown"],
        "Ship type": ["Cargo", "Fishing"],
        "Cargo type": [70.0, 30.0],
        "Width": [20.0, 10.0],
        "Length": [200.0, 50.0],
        "Type of position fixing device": ["GPS", "GPS"],
        "Draught": [10.0, 5.0],
        "Destination": ["COPENHAGEN", "Unknown"],
        "ETA": ["02/03/2024 12:00:00", "Unknown"],
        "Data source type": ["AIS", "AIS"],
        "A": [10.0, 5.0],
        "B": [190.0, 45.0],
        "C": [5.0, 3.0],
        "D": [5.0, 3.0],
    })
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def loader(tmp_path):
    """Create a loader instance pointing to test data."""
    return AISDKLoader(data_root=str(tmp_path))


class TestAISDKLoader:
    """Test cases for AISDKLoader."""

    def test_init(self, tmp_path):
        """Test loader initialization."""
        loader = AISDKLoader(data_root=str(tmp_path))
        assert loader.root == Path(tmp_path)
        assert loader.date_start is None
        assert loader.date_end is None

    def test_init_with_dates(self, tmp_path):
        """Test loader initialization with date filters."""
        loader = AISDKLoader(
            data_root=str(tmp_path),
            date_start="2024-03-01",
            date_end="2024-03-31",
        )
        assert loader.date_start == pd.Timestamp("2024-03-01")
        assert loader.date_end == pd.Timestamp("2024-03-31")

    def test_get_files(self, sample_csv_path):
        """Test file discovery."""
        tmp_path = sample_csv_path.parent.parent
        loader = AISDKLoader(data_root=str(tmp_path))
        files = loader._get_files()
        assert len(files) == 1
        assert "aisdk-2024-03-01.csv" in str(files[0])

    def test_get_files_with_date_filter(self, sample_csv_path):
        """Test file discovery with date filtering."""
        tmp_path = sample_csv_path.parent.parent
        loader = AISDKLoader(
            data_root=str(tmp_path),
            date_start="2024-03-01",
            date_end="2024-03-01",
        )
        files = loader._get_files()
        assert len(files) == 1

    def test_read_file(self, sample_csv_path):
        """Test reading a single CSV file."""
        loader = AISDKLoader()
        df = loader._read_file(sample_csv_path)
        
        assert len(df) == 2
        assert "timestamp" in df.columns
        assert "mmsi" in df.columns
        assert "lat" in df.columns
        assert "lon" in df.columns
        
        # Check column rename
        assert "# Timestamp" not in df.columns
        
        # Check timestamp parsing
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_load(self, sample_csv_path):
        """Test full load method."""
        tmp_path = sample_csv_path.parent.parent
        loader = AISDKLoader(data_root=str(tmp_path))
        df = loader.load()
        
        assert len(df) == 2
        assert "timestamp" in df.columns
        assert "source_date" in df.columns

    def test_load_no_files(self, tmp_path):
        """Test load with no matching files."""
        loader = AISDKLoader(data_root=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            loader.load()

    def test_column_rename_map(self):
        """Test that all expected columns are in the rename map."""
        expected_columns = [
            "# Timestamp", "Type of mobile", "MMSI", "Latitude", "Longitude",
            "Navigational status", "ROT", "SOG", "COG", "Heading", "IMO",
            "Callsign", "Name", "Ship type", "Cargo type", "Width", "Length",
            "Type of position fixing device", "Draught", "Destination", "ETA",
            "Data source type", "A", "B", "C", "D"
        ]
        for col in expected_columns:
            assert col in COLUMN_RENAME

    def test_dtype_map(self):
        """Test that dtype map has correct types."""
        assert DTYPE_MAP["MMSI"] == "int64"
        assert DTYPE_MAP["Latitude"] == "float32"
        assert DTYPE_MAP["Longitude"] == "float32"


class TestAISDKLoaderIntegration:
    """Integration tests for loader."""

    def test_load_multiple_days(self, tmp_path):
        """Test loading multiple days of data."""
        # Create multiple day directories
        for day in ["2024-03-01", "2024-03-02"]:
            day_dir = tmp_path / day
            day_dir.mkdir()
            df = pd.DataFrame({
                "# Timestamp": [f"01/03/2024 00:00:00"],
                "Type of mobile": ["Class A"],
                "MMSI": [123456789],
                "Latitude": [56.0],
                "Longitude": [12.0],
                "Navigational status": ["Under way using engine"],
                "ROT": [0.0],
                "SOG": [10.0],
                "COG": [180.0],
                "Heading": [180],
                "IMO": ["1234567"],
                "Callsign": ["ABC123"],
                "Name": ["VESSEL_1"],
                "Ship type": ["Cargo"],
                "Cargo type": [70.0],
                "Width": [20.0],
                "Length": [200.0],
                "Type of position fixing device": ["GPS"],
                "Draught": [10.0],
                "Destination": ["COPENHAGEN"],
                "ETA": ["02/03/2024 12:00:00"],
                "Data source type": ["AIS"],
                "A": [10.0],
                "B": [190.0],
                "C": [5.0],
                "D": [5.0],
            })
            df.to_csv(day_dir / f"aisdk-{day}.csv", index=False)
        
        loader = AISDKLoader(data_root=str(tmp_path))
        df = loader.load()
        
        assert len(df) == 2

    def test_source_date_injection(self, sample_csv_path):
        """Test that source_date is correctly injected."""
        loader = AISDKLoader()
        df = loader._read_file(sample_csv_path)
        
        assert "source_date" in df.columns
        assert df["source_date"].iloc[0] == pd.Timestamp("2024-03-01").date()
