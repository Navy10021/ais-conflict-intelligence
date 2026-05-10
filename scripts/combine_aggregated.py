"""
Combine Aggregated Data with Conflict Labels
=============================================
21일치 aggregated 파일을 합치고, features 파일에서
conflict_label을 추출해 병합한다.

실행:
    python scripts/combine_aggregated.py
"""
import pandas as pd
import glob
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


def combine_aggregated_with_labels(
    agg_pattern: str = "outputs/processed/ais_aggregated_2022-*.parquet",
    feat_pattern: str = "outputs/processed/ais_features_*.parquet",
    output_path: str = "outputs/processed/ais_aggregated_combined.parquet",
) -> pd.DataFrame:
    """
    Aggregated 파일들을 합치고 conflict_label을 병합한다.
    """
    # 1. Aggregated 파일 합치기
    agg_files = sorted(glob.glob(agg_pattern))
    logger.info(f"Aggregated files found: {len(agg_files)}")
    agg = pd.concat([pd.read_parquet(f) for f in agg_files], ignore_index=True)
    logger.info(f"Combined aggregated: {len(agg):,} rows")

    # 2. Features 파일에서 conflict 라벨 추출
    feat_files = sorted(glob.glob(feat_pattern))
    logger.info(f"Features files found: {len(feat_files)}")

    conflict_dfs = []
    for f in feat_files:
        df = pd.read_parquet(
            f,
            columns=[
                "time_bucket", "grid_cell",
                "conflict_label", "conflict_intensity", "days_to_conflict",
                "conflict_zone_name", "in_conflict_zone", "is_control_zone",
            ]
        )
        cl = df.groupby(["grid_cell", "time_bucket"]).agg(
            conflict_label=("conflict_label", "max"),
            conflict_intensity=("conflict_intensity", "max"),
            days_to_conflict=("days_to_conflict", "min"),
            conflict_zone_name=("conflict_zone_name", "first"),
            in_conflict_zone=("in_conflict_zone", "max"),
            is_control_zone=("is_control_zone", "max"),
        ).reset_index()
        conflict_dfs.append(cl)
        logger.info(f"  {Path(f).name}: {len(cl)} rows")

    conflict_all = pd.concat(conflict_dfs, ignore_index=True)
    logger.info(f"Conflict labels combined: {len(conflict_all):,} rows")

    # 3. Aggregated + conflict_label 병합
    merged = agg.merge(
        conflict_all,
        on=["grid_cell", "time_bucket"],
        how="left"
    )
    logger.info(f"Merged: {len(merged):,} rows")

    # 4. timestamp 컬럼 추가 (event_study, ITS 분석용)
    merged["timestamp"] = pd.to_datetime(merged["time_bucket"])
    if merged["timestamp"].dt.tz is not None:
        merged["timestamp"] = merged["timestamp"].dt.tz_localize(None)

    # 5. 저장
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    logger.info(f"Saved to {output_path}")
    logger.info(f"Final shape: {merged.shape}")
    logger.info(f"Columns: {merged.columns.tolist()}")

    return merged


if __name__ == "__main__":
    combine_aggregated_with_labels()