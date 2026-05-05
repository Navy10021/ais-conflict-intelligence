#!/bin/bash
# MCIS Full Pipeline Script
# Runs the entire Maritime Conflict Intelligence System pipeline

set -e  # Exit on error

echo "========================================="
echo "MCIS Pipeline Start"
echo "========================================="

# Configuration
DATA_ROOT="./data"
DATE_START="2024-03-01"
DATE_END="2024-08-31"
OUTPUT_DIR="./outputs"
PROCESSED_DIR="${OUTPUT_DIR}/processed"
FIGURES_DIR="${OUTPUT_DIR}/figures"
TABLES_DIR="${OUTPUT_DIR}/tables"
MODELS_DIR="${OUTPUT_DIR}/models"

# Create output directories
mkdir -p "${PROCESSED_DIR}" "${FIGURES_DIR}" "${TABLES_DIR}" "${MODELS_DIR}"

# Step 1: Load data
echo ""
echo "[1/7] Loading AIS data..."
python -m src.preprocessing.loader \
    --data-root "${DATA_ROOT}" \
    --date-start "${DATE_START}" \
    --date-end "${DATE_END}" \
    --output "${PROCESSED_DIR}/ais_raw.parquet" \
    --verbose

# Step 2: Clean data
echo ""
echo "[2/7] Cleaning AIS data..."
python -m src.preprocessing.cleaner \
    --input "${PROCESSED_DIR}/ais_raw.parquet" \
    --output "${PROCESSED_DIR}/ais_clean.parquet" \
    --verbose

# Step 3: Feature engineering
echo ""
echo "[3/7] Generating features..."
python -m src.preprocessing.feature_engineer \
    --input "${PROCESSED_DIR}/ais_clean.parquet" \
    --output "${PROCESSED_DIR}/ais_features.parquet" \
    --conflict-events "data_external/acled_events.csv" \
    --ports "data_external/world_ports.csv" \
    --verbose

# Step 4: Aggregation
echo ""
echo "[4/7] Aggregating data..."
python -m src.preprocessing.aggregator \
    --input "${PROCESSED_DIR}/ais_features.parquet" \
    --output "${PROCESSED_DIR}/ais_aggregated.parquet" \
    --verbose

# Step 5: Visualization
echo ""
echo "[5/7] Generating visualizations..."
python -m src.visualization.spatial_viz \
    --input "${PROCESSED_DIR}/ais_features.parquet" \
    --output-dir "${FIGURES_DIR}/spatial" \
    --verbose &

python -m src.visualization.temporal_viz \
    --input "${PROCESSED_DIR}/ais_features.parquet" \
    --output-dir "${FIGURES_DIR}/temporal" \
    --verbose &

python -m src.visualization.statistical_viz \
    --input "${PROCESSED_DIR}/ais_features.parquet" \
    --output-dir "${FIGURES_DIR}/statistical" \
    --verbose &

wait  # Wait for background processes

# Step 6: Analysis
echo ""
echo "[6/7] Running correlation analysis..."
python -m src.analysis.correlation_analyzer \
    --input "${PROCESSED_DIR}/ais_features.parquet" \
    --output "${TABLES_DIR}" \
    --verbose &

python -m src.analysis.traffic_analyzer \
    --input "${PROCESSED_DIR}/ais_features.parquet" \
    --output "${TABLES_DIR}" \
    --verbose &

python -m src.analysis.behavioral_analyzer \
    --input "${PROCESSED_DIR}/ais_features.parquet" \
    --output "${TABLES_DIR}" \
    --verbose &

python -m src.analysis.destination_analyzer \
    --input "${PROCESSED_DIR}/ais_features.parquet" \
    --output "${TABLES_DIR}" \
    --verbose &

python -m src.analysis.rot_analyzer \
    --input "${PROCESSED_DIR}/ais_features.parquet" \
    --output "${TABLES_DIR}" \
    --verbose &

wait

# Step 7: Model training
echo ""
echo "[7/7] Training models..."
python -m src.models.anomaly_model \
    --input "${PROCESSED_DIR}/ais_features.parquet" \
    --output "${MODELS_DIR}/anomaly" \
    --verbose &

python -m src.models.conflict_predictor \
    --input "${PROCESSED_DIR}/ais_aggregated.parquet" \
    --mode train \
    --output "${MODELS_DIR}/predictor" \
    --verbose &

python -m src.models.baseline \
    --input "${PROCESSED_DIR}/ais_aggregated.parquet" \
    --output "${MODELS_DIR}/baseline" \
    --verbose &

wait

echo ""
echo "========================================="
echo "MCIS Pipeline Complete!"
echo "========================================="
echo "Outputs:"
echo "  - Processed data: ${PROCESSED_DIR}"
echo "  - Figures: ${FIGURES_DIR}"
echo "  - Tables: ${TABLES_DIR}"
echo "  - Models: ${MODELS_DIR}"
