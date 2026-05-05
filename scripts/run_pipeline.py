"""
MCIS Full Pipeline Script (Cross-Platform)
Runs the entire Maritime Conflict Intelligence System pipeline
"""
import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration
DATA_ROOT = "./data"
DATE_START = "2024-03-01"
DATE_END = "2024-08-31"
OUTPUT_DIR = "./outputs"
PROCESSED_DIR = f"{OUTPUT_DIR}/processed"
FIGURES_DIR = f"{OUTPUT_DIR}/figures"
TABLES_DIR = f"{OUTPUT_DIR}/tables"
MODELS_DIR = f"{OUTPUT_DIR}/models"


def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return success status."""
    logger.info("\n{'='*60}")
    logger.info("[{description}]")
    logger.info("{'='*60}")
    logger.info("Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )
        logger.info("[SUCCESS] %s completed successfully", description)
        return True
    except subprocess.CalledProcessError as e:
        logger.error("[FAILED] %s failed with exit code %d", description, e.returncode)
        return False


def create_output_dirs():
    """Create necessary output directories."""
    dirs = [PROCESSED_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        logger.info("Created directory: %s", d)


def run_preprocessing(verbose: bool = False) -> bool:
    """Run preprocessing steps (load, clean, feature engineering, aggregate)."""
    v = ["--verbose"] if verbose else []
    
    # Step 1: Load data
    if not run_command(
        [sys.executable, "-m", "src.preprocessing.loader",
         "--data-root", DATA_ROOT,
         "--date-start", DATE_START,
         "--date-end", DATE_END,
         "--output", f"{PROCESSED_DIR}/ais_raw.parquet"] + v,
        "1/7 Loading AIS data"
    ):
        return False
    
    # Step 2: Clean data
    if not run_command(
        [sys.executable, "-m", "src.preprocessing.cleaner",
         "--input", f"{PROCESSED_DIR}/ais_raw.parquet",
         "--output", f"{PROCESSED_DIR}/ais_clean.parquet"] + v,
        "2/7 Cleaning AIS data"
    ):
        return False
    
    # Step 3: Feature engineering
    if not run_command(
        [sys.executable, "-m", "src.preprocessing.feature_engineer",
         "--input", f"{PROCESSED_DIR}/ais_clean.parquet",
         "--output", f"{PROCESSED_DIR}/ais_features.parquet",
         "--conflict-events", "data_external/acled_events.csv",
         "--ports", "data_external/world_ports.csv"] + v,
        "3/7 Generating features"
    ):
        return False
    
    # Step 4: Aggregation
    if not run_command(
        [sys.executable, "-m", "src.preprocessing.aggregator",
         "--input", f"{PROCESSED_DIR}/ais_features.parquet",
         "--output", f"{PROCESSED_DIR}/ais_aggregated.parquet"] + v,
        "4/7 Aggregating data"
    ):
        return False
    
    return True


def run_visualization(verbose: bool = False) -> bool:
    """Run visualization modules."""
    v = ["--verbose"] if verbose else []
    
    logger.info("\n{'='*60}")
    logger.info("[5/7] Generating visualizations")
    logger.info("{'='*60}")
    
    commands = [
        ([sys.executable, "-m", "src.visualization.spatial_viz",
         "--input", f"{PROCESSED_DIR}/ais_features.parquet",
         "--output-dir", f"{FIGURES_DIR}/spatial"] + v,
         "Spatial visualization"),
        
        ([sys.executable, "-m", "src.visualization.temporal_viz",
         "--input", f"{PROCESSED_DIR}/ais_features.parquet",
         "--output-dir", f"{FIGURES_DIR}/temporal"] + v,
         "Temporal visualization"),
        
        ([sys.executable, "-m", "src.visualization.statistical_viz",
         "--input", f"{PROCESSED_DIR}/ais_features.parquet",
         "--output-dir", f"{FIGURES_DIR}/statistical"] + v,
         "Statistical visualization"),
    ]
    
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            logger.warning("Warning: {desc} failed, continuing...")
    
    return True


def run_analysis(verbose: bool = False) -> bool:
    """Run analysis modules."""
    v = ["--verbose"] if verbose else []
    
    logger.info("\n{'='*60}")
    logger.info("[6/7] Running correlation analysis")
    logger.info("{'='*60}")
    
    commands = [
        ([sys.executable, "-m", "src.analysis.correlation_analyzer",
         "--input", f"{PROCESSED_DIR}/ais_features.parquet",
         "--output", TABLES_DIR] + v,
         "Correlation analysis"),
        
        ([sys.executable, "-m", "src.analysis.traffic_analyzer",
         "--input", f"{PROCESSED_DIR}/ais_features.parquet",
         "--output", TABLES_DIR] + v,
         "Traffic analysis"),
        
        ([sys.executable, "-m", "src.analysis.behavioral_analyzer",
         "--input", f"{PROCESSED_DIR}/ais_features.parquet",
         "--output", TABLES_DIR] + v,
         "Behavioral analysis"),
        
        ([sys.executable, "-m", "src.analysis.destination_analyzer",
         "--input", f"{PROCESSED_DIR}/ais_features.parquet",
         "--output", TABLES_DIR] + v,
         "Destination analysis"),
        
        ([sys.executable, "-m", "src.analysis.rot_analyzer",
         "--input", f"{PROCESSED_DIR}/ais_features.parquet",
         "--output", TABLES_DIR] + v,
         "ROT analysis"),
    ]
    
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            logger.warning("Warning: {desc} failed, continuing...")
    
    return True


def run_models(verbose: bool = False) -> bool:
    """Run model training."""
    v = ["--verbose"] if verbose else []
    
    logger.info("\n{'='*60}")
    logger.info("[7/7] Training models")
    logger.info("{'='*60}")
    
    commands = [
        ([sys.executable, "-m", "src.models.anomaly_model",
         "--input", f"{PROCESSED_DIR}/ais_features.parquet",
         "--output", f"{MODELS_DIR}/anomaly"] + v,
         "Anomaly detection model"),
        
        ([sys.executable, "-m", "src.models.conflict_predictor",
         "--input", f"{PROCESSED_DIR}/ais_aggregated.parquet",
         "--mode", "train",
         "--output", f"{MODELS_DIR}/predictor"] + v,
         "Conflict predictor model"),
        
        ([sys.executable, "-m", "src.models.baseline",
         "--input", f"{PROCESSED_DIR}/ais_aggregated.parquet",
         "--output", f"{MODELS_DIR}/baseline"] + v,
         "Baseline models"),
    ]
    
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            logger.warning("Warning: {desc} failed, continuing...")
    
    return True


def main():
    global DATE_START, DATE_END, DATA_ROOT
    
    parser = argparse.ArgumentParser(
        description="Run the full MCIS pipeline"
    )
    parser.add_argument(
        "--skip-preprocessing", action="store_true",
        help="Skip preprocessing steps"
    )
    parser.add_argument(
        "--skip-viz", action="store_true",
        help="Skip visualization"
    )
    parser.add_argument(
        "--skip-analysis", action="store_true",
        help="Skip analysis"
    )
    parser.add_argument(
        "--skip-models", action="store_true",
        help="Skip model training"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Verbose logging"
    )
    parser.add_argument(
        "--date-start", default=DATE_START,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--date-end", default=DATE_END,
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--data-root", default=DATA_ROOT,
        help="Path to data directory"
    )
    
    args = parser.parse_args()
    
    # Update globals from args
    DATE_START = args.date_start
    DATE_END = args.date_end
    DATA_ROOT = args.data_root
    
    logger.info("="*60)
    logger.info("MCIS Pipeline Start")
    logger.info("="*60)
    logger.info("Date range: {DATE_START} to {DATE_END}")
    logger.info("Data root: {DATA_ROOT}")
    logger.info("Output dir: {OUTPUT_DIR}")
    
    # Create output directories
    create_output_dirs()
    
    success = True
    
    # Run pipeline steps
    if not args.skip_preprocessing:
        if not run_preprocessing(verbose=args.verbose):
            logger.error("Preprocessing failed!")
            success = False
    
    if not args.skip_viz and success:
        run_visualization(verbose=args.verbose)
    
    if not args.skip_analysis and success:
        run_analysis(verbose=args.verbose)
    
    if not args.skip_models and success:
        run_models(verbose=args.verbose)
    
    logger.info("\n" + "="*60)
    if success:
        logger.info("MCIS Pipeline Complete!")
    else:
        logger.warning("MCIS Pipeline completed with errors!")
    logger.info("="*60)
    logger.info("Outputs:")
    logger.info("  - Processed data: {PROCESSED_DIR}")
    logger.info("  - Figures: {FIGURES_DIR}")
    logger.info("  - Tables: {TABLES_DIR}")
    logger.info("  - Models: {MODELS_DIR}")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

