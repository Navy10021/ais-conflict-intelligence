# AIS Conflict Intelligence Codebase Analysis Report

## Executive Summary

This repository presents a well-structured end-to-end prototype for maritime conflict intelligence using AIS data, with clear module separation across preprocessing, analysis, visualization, and modeling. The design is research-oriented and pragmatic, but there are several engineering risks (test fragility, logging/encoding quality, dependency heaviness, and package-layout quirks) that should be addressed before production-scale use.

## Scope and Method

I reviewed:

- Project metadata and architecture definitions (`README.md`, `pyproject.toml`, `CLAUDE.md`).
- Core preprocessing/feature-generation pipeline modules.
- Test execution behavior in the current environment.

## Architecture Assessment

### Strengths

1. **Clear domain decomposition**
   - `src/preprocessing`, `src/analysis`, `src/visualization`, and `src/models` provide understandable boundaries for the analytics lifecycle.
2. **Pipeline-first workflow**
   - CLI-capable modules (`loader.py`, `cleaner.py`, `feature_engineer.py`) make batch execution straightforward.
3. **Research signal richness**
   - Feature engineering includes temporal, geospatial, behavioral, and conflict-context features.
4. **Configurable package metadata**
   - `pyproject.toml` includes dependency and linting/test tooling definitions.

### Risks / Gaps

1. **Mixed maturity between research code and library code**
   - Some modules prioritize experimentation over robust API contracts.
2. **Large dependency footprint**
   - Heavy ML/time-series stack increases setup complexity and CI fragility.
3. **Potential source-layout packaging mismatch**
   - `setuptools` `where = ["src"]` with `include = ["src*"]` may be unconventional for downstream packaging expectations.

## Code Quality Findings

## 1) Preprocessing Loader (`src/preprocessing/loader.py`)

### Positives
- Good column normalization strategy with rename map.
- Date-range file filtering is intuitive.
- Timestamp parsing is explicit and UTC-aware.

### Improvement Opportunities
- `chunksize` is defined but not actively used for chunked `read_csv` streaming in `_read_file`; this can mislead users expecting low-memory reads.
- Logging statements use f-string-like placeholders in plain strings in a few places elsewhere in project; standard `%s` formatting should be consistent.

## 2) Cleaner (`src/preprocessing/cleaner.py`)

### Positives
- Sensible cleaning pipeline ordering by likely computational cost.
- Domain-aware sentinel-value handling for SOG/COG/Heading/ROT.
- MMSI and coordinate validation are explicit.

### Improvement Opportunities
- Several log lines use literal braces (e.g., `"{n - len(df):,}"`) without f-string or logger args, reducing observability quality.
- File has visible encoding artifacts in comments/docstrings (garbled characters), which reduces maintainability and documentation trust.
- Hard-coded thresholds are practical but should migrate to config for experiment reproducibility.

## 3) Feature Engineering (`src/preprocessing/feature_engineer.py`)

### Positives
- Rich, domain-informed features (dark ship, ROT anomalies, conflict-zone flags, route entropy, destination dynamics).
- Dynamic aggregation dictionary is a good extensibility design.

### Improvement Opportunities
- `add_destination_features()` assumes a specific external schema (`conflict_zone`, `locode`) without schema validation fallback.
- Several methods rely on earlier stage columns (ordering coupling); explicit precondition checks could prevent subtle runtime errors.
- Conflict zones are bbox-based; polygon-based geofencing would reduce false positives at zone boundaries.

## Testing and Reliability Assessment

### Current test-state findings

1. Running `pytest -q` fails because `pytest-cov` options are configured in `pyproject.toml` but plugin availability is missing in this environment.
2. Running tests without `addopts` (`python -m pytest tests -q -o addopts=''`) fails during collection because `tests/test_aggregator.py` executes pipeline code at import time and expects a local artifact `outputs/processed/sample_clean.parquet`.

### Reliability implications

- Test suite is not hermetic: it depends on local files not provisioned by fixtures.
- At least one test module has side effects at import-time, which is a maintainability anti-pattern.

## Priority Recommendations

## P0 (Immediate)

1. **Fix test hermeticity**
   - Replace import-time execution with fixture-driven setup.
   - Generate required parquet fixtures in `tmp_path`.
2. **Standardize logging format calls**
   - Replace literal brace strings with logger argument formatting.
3. **Repair source text encoding artifacts**
   - Normalize files to UTF-8 and clean damaged docstrings/comments.

## P1 (Next)

1. **Make thresholds configurable**
   - Move sentinel thresholds and business rules to `config/settings.yaml`.
2. **Add validation contracts**
   - Implement column precondition checks for each transformation stage.
3. **Use true chunk streaming in loader**
   - Honor `chunksize` via iterative `read_csv(..., chunksize=...)` path.

## P2 (Strategic)

1. **Package layout hardening**
   - Revisit `pyproject.toml` package discovery to align with standard `src/` layout expectations.
2. **Dependency profile split**
   - Separate runtime-minimal install from research-heavy extras (e.g., `ml`, `viz`, `forecasting`).
3. **Geospatial precision upgrade**
   - Move from bbox filters to polygon spatial joins for zone assignment.

## Suggested Near-Term Roadmap (2 weeks)

- **Day 1-2**: Test refactor for hermetic CI + logging fixes.
- **Day 3-5**: Config-driven thresholds + validation guards.
- **Day 6-8**: Loader chunked-stream implementation + benchmarks.
- **Day 9-10**: Packaging/dependency split and docs refresh.

## Conclusion

The repository already has strong research-domain foundations and modular structure. By prioritizing test hermeticity, logging/encoding cleanup, and config-driven behavior, the project can quickly improve reproducibility and operational reliability while preserving analytical depth.
